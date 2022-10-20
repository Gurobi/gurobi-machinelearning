# Copyright © 2022 Gurobi Optimization, LLC
""" Module for insterting an :external+sklearn:py:class:`sklearn.pipeline.Pipeline` into a gurobipy model
"""


from ..exceptions import NoModel
from ..modeling import AbstractPredictorConstr
from .list import sklearn_predictors, sklearn_transformers
from .skgetter import SKgetter


class PipelineConstr(SKgetter, AbstractPredictorConstr):
    """Use a scikit-learn pipeline to build constraints in Gurobi model."""

    def __init__(self, grbmodel, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self._kwargs = kwargs
        SKgetter.__init__(self, pipeline)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def _mip_model(self):
        pipeline = self.predictor
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
                raise NoModel(pipeline, f"I don't know how to deal with that object: {name}")
            input_vars = steps[-1].output
        name, obj = pipeline.steps[-1]
        predictors = {}
        for key, item in sklearn_predictors().items():
            predictors[key.lower()] = item
        try:
            steps.append(predictors[name](model, obj, input_vars, output_vars, **self._kwargs))
        except KeyError:
            raise NoModel(pipeline, f"I don't know how to deal with that object: {name}")
        if self._output is None:
            self._output = steps[-1].output

    def print_stats(self, file=None):
        """Print statistics on model additions stored by this class

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        Usually derived classes reimplement this function to provide more
        details about the structure of the additions (type of ML model,
        layers if it's a neural network,...)

        Arguments
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(file=file)
        print(file=file)
        print(f"Pipeline has {len(self._steps)} steps:", file=file)
        for step in self:
            print(step, end=" ", file=file)
        print(file=file)
        print(file=file)
        for step in self:
            step.print_stats(file)
            print(file=file)

    def __getitem__(self, key):
        """Get an item from the pipeline steps"""
        return self._steps[key]

    def __iter__(self):
        """Iterate through pipeline steps"""
        return self._steps.__iter__()

    def __len__(self):
        """Get number of pipeline steps"""
        return self._steps.__len__()


def add_pipeline_constr(grbmodel, pipeline, input_vars, output_vars=None, **kwargs):
    """Use a `pipeline` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    pipeline: :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
        The pipeline to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    PipelineConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Raises
    ------
    NoModel
        If the translation to Gurobi of one of the elements in the pipeline
        is not implemented or recognized.

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return PipelineConstr(grbmodel, pipeline, input_vars, output_vars, **kwargs)
