# Copyright © 2022 Gurobi Optimization, LLC
""" A set of classes for transforming Scikit-Learn regression objects
to constraints in Gurobi

What we have so far:
  - LinearRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - LogisticRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - MLSRegressionConstr: a neural network.
"""


import numpy as np

from ..basepredictor import AbstractPredictorConstr, BaseNNConstr, _default_name


class LinearRegressionConstr(BaseNNConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, **kwargs):
        super().__init__(grbmodel, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.addlayer(
            self._input,
            self.regressor.coef_.T.reshape(-1, 1),
            np.array(self.regressor.intercept_).reshape((-1,)),
            self.actdict["identity"],
            self._output,
        )
        if self._output is None:
            self._output = self._layers[-1].output


class LogisticRegressionConstr(BaseNNConstr):
    """Predict a Gurobi variable using a Logistic Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, **kwargs):
        super().__init__(grbmodel, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.addlayer(
            self._input,
            self.regressor.coef_.T,
            self.regressor.intercept_,
            self.actdict["logit"],
            self._output,
        )
        if self._output is None:
            self._output = self._layers[-1].output


class MLPRegressorConstr(BaseNNConstr):
    """Predict a Gurobi matrix variable using a neural network that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, clean_regressor=False, **kwargs):
        super().__init__(
            grbmodel,
            regressor,
            input_vars,
            output_vars,
            clean_regressor=clean_regressor,
            **kwargs,
        )
        assert regressor.out_activation_ in ("identity", "relu")

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        neuralnet = self.regressor
        if neuralnet.activation not in self.actdict:
            print(self.actdict)
            raise BaseException(f"No implementation for activation function {neuralnet.activation}")
        activation = self.actdict[neuralnet.activation]

        input_vars = self._input
        output = None

        for i in range(neuralnet.n_layers_ - 1):
            layer_coefs = neuralnet.coefs_[i]
            layer_intercept = neuralnet.intercepts_[i]

            # For last layer change activation
            if i == neuralnet.n_layers_ - 2:
                activation = self.actdict[neuralnet.out_activation_]
                output = self._output

            layer = self.addlayer(input_vars, layer_coefs, layer_intercept, activation, output, name=f"layer{i}")
            input_vars = layer._output  # pylint: disable=W0212
            self._model.update()
        if self._output is None:
            self._output = layer.output


class PipelineConstr(AbstractPredictorConstr):
    """Use a scikit-learn pipeline to build constraints in Gurobi model."""

    def __init__(self, grbmodel, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self.pipeline = pipeline
        self._kwargs = kwargs
        super().__init__(grbmodel, input_vars, output_vars, default_name=_default_name(pipeline), **kwargs)

    def mip_model(self):
        pipeline = self.pipeline
        model = self._model
        input_vars = self._input
        output_vars = self._output
        steps = self._steps
        transformers = {}
        for key, item in sklearn_transformers().items():
            transformers[key.lower()] = item
        for name, obj in pipeline.steps[:-1]:
            try:
                steps.append(transformers[name](model, obj, input_vars, **self._kwargs))
            except KeyError:
                raise BaseException(f"I don't know how to deal with that object: {name}")
            input_vars = steps[-1].output
        name, obj = pipeline.steps[-1]
        predictors = {}
        for key, item in sklearn_predictors().items():
            predictors[key.lower()] = item
        try:
            steps.append(predictors[name](model, obj, input_vars, output_vars, **self._kwargs))
        except KeyError:
            raise BaseException(f"I don't know how to deal with that object: {name}")
        if self._output is None:
            self._output = steps[-1].output

    def print_stats(self, file=None):
        super().print_stats(file=file)
        print()
        print(f"Pipeline has {len(self._steps)} steps", file=file)
        for step in self:
            step.print_stats(file)
            print()

    def __getitem__(self, key):
        """Get an item from the pipeline steps"""
        return self._steps[key]

    def __iter__(self):
        """Iterate through pipeline steps"""
        return self._steps.__iter__()

    def __len__(self):
        """Get number of pipeline steps"""
        return self._steps.__len__()
