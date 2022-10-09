""" Implementation for the scikit learn pipeline """


from ..base import AbstractPredictorConstr
from .list import sklearn_predictors, sklearn_transformers
from .skgetter import SKgetter


class PipelineConstr(SKgetter, AbstractPredictorConstr):
    """Use a scikit-learn pipeline to build constraints in Gurobi model."""

    def __init__(self, grbmodel, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self._kwargs = kwargs
        SKgetter.__init__(self, pipeline)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
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
        print(f"Pipeline has {len(self._steps)} steps:", file=file)
        for step in self:
            print(step, end=" ", file=file)
        print()
        print()
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
