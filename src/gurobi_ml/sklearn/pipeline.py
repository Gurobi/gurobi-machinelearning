# Copyright © 2022 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Module for embedding a :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
into a :gurobipy:`model`.
"""


from ..exceptions import NoModel
from ..modeling.basepredictor import AbstractPredictorConstr
from ..register_predictor import user_predictors
from .predictors_list import sklearn_predictors, sklearn_transformers
from .skgetter import SKgetter


def add_pipeline_constr(gp_model, pipeline, input_vars, output_vars=None, **kwargs):
    """Embed pipeline into gp_model

    Predict the values of output_vars using input_vars

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    pipeline: :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
        The pipeline to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for regression in model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for regression in model.

    Returns
    -------
    PipelineConstr
        Object containing information about what was added to gp_model to embed the
        predictor into it

    Raises
    ------
    NoModel
        If the translation to Gurobi of one of the elements in the pipeline
        is not implemented or recognized.

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return PipelineConstr(gp_model, pipeline, input_vars, output_vars, **kwargs)


class PipelineConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.pipeline.Pipeline` with gurobipy

    Stores the changes to :gurobipy:`model` when embedding an instance into it."""

    def __init__(self, gp_model, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self._default_name = "pipe"
        SKgetter.__init__(self, pipeline, **kwargs)
        AbstractPredictorConstr.__init__(self, gp_model, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        pipeline = self.predictor
        gp_model = self._gp_model
        input_vars = self._input
        output_vars = self._output
        steps = self._steps
        transformers = {}
        for key, item in sklearn_transformers().items():
            transformers[key.lower()] = item
        for name, obj in pipeline.steps[:-1]:
            try:
                steps.append(transformers[name](gp_model, obj, input_vars, **kwargs))
            except KeyError:
                raise NoModel(pipeline, f"I don't know how to deal with that object: {name}")
            input_vars = steps[-1].output
        name, obj = pipeline.steps[-1]
        predictors = {}
        for key, item in sklearn_predictors().items():
            predictors[key.lower()] = item
        for key, item in user_predictors().items():
            if not isinstance(key, str):
                key = key.__name__
            predictors[key.lower()] = item
        try:
            steps.append(predictors[name](gp_model, obj, input_vars, output_vars, **kwargs))
        except KeyError:
            raise NoModel(pipeline, f"I don't know how to deal with that object: {name}")
        if self._output is None:
            self._output = steps[-1].output

    def print_stats(self, file=None):
        """Print statistics on model additions stored by this class

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        The pipeline version includes a summary of the steps that it contains.

        Arguments
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(file=file)
        print(file=file)
        print(f"Pipeline has {len(self._steps)} steps:", file=file)
        print(file=file)

        header = f"{'Step':13} {'Output Shape':>14} {'Variables':>12} {'Constraints':^38}"
        print("-" * len(header), file=file)
        print(header, file=file)
        print(f"{' '*41} {'Linear':>12} {'Quadratic':>12} {'General':>12}", file=file)
        print("=" * len(header), file=file)
        for step in self:
            step.print_stats(abbrev=True, file=file)
            print(file=file)
        print("-" * len(header), file=file)

    def __getitem__(self, key):
        """Get an item from the pipeline steps"""
        return self._steps[key]

    def __iter__(self):
        """Iterate through pipeline steps"""
        return self._steps.__iter__()

    def __len__(self):
        """Get number of pipeline steps"""
        return self._steps.__len__()
