# Copyright © 2023 Gurobi Optimization, LLC
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

"""Module for formulating a :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
in a :gurobipy:`model`.
"""


from ..exceptions import NoModel
from ..modeling.base_predictor_constr import AbstractPredictorConstr
from ..modeling.get_convertor import get_convertor
from ..register_user_predictor import user_predictors
from ..xgboost_sklearn_api import xgboost_sklearn_convertors
from .column_transformer import add_column_transformer_constr
from .predictors_list import sklearn_predictors
from .preprocessing import sklearn_transformers
from .skgetter import SKgetter


def add_pipeline_constr(gp_model, pipeline, input_vars, output_vars=None, **kwargs):
    """Formulate pipeline into gp_model.

    The formulation predicts the values of output_vars using input_vars according to
    pipeline.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    pipeline : :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
        The pipeline to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for regression in model.
    output_vars : mvar_array_like, optional
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

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return PipelineConstr(gp_model, pipeline, input_vars, output_vars, **kwargs)


class PipelineConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
    in a gurobipy model.

    |ClassShort|
    """

    def __init__(self, gp_model, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self._default_name = "pipe"
        SKgetter.__init__(self, pipeline, input_vars, **kwargs)
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, validate_input=False, **kwargs
        )

    def _build_submodel(self, gp_model, *args, **kwargs):
        """Predict output from input using predictor or transformer.

        Pipelines are different from other objects because they can't validate
        their input and output. They are just containers of other objects that will
        do it.
        """
        self._mip_model(**kwargs)
        assert self.output is not None
        assert self.input is not None
        # We can call validate only after the model is created
        self._validate()
        return self

    def _mip_model(self, **kwargs):
        pipeline = self.predictor
        gp_model = self.gp_model
        input_vars = self._input
        output_vars = self._output
        steps = self._steps
        transformers = sklearn_transformers()
        transformers |= user_predictors()
        transformers["ColumnTransformer"] = add_column_transformer_constr
        kwargs["validate_input"] = True

        for transformer in pipeline[:-1]:
            convertor = get_convertor(transformer, transformers)
            if convertor is None:
                raise NoModel(
                    self.predictor,
                    f"I don't know how to deal with that object: {transformer}",
                )
            steps.append(convertor(gp_model, transformer, input_vars, **kwargs))
            input_vars = steps[-1].output

        predictor = pipeline[-1]
        predictors = sklearn_predictors() | user_predictors()
        predictors |= xgboost_sklearn_convertors()
        convertor = get_convertor(predictor, predictors)
        if convertor is None:
            raise NoModel(
                self.predictor,
                f"I don't know how to deal with that object: {predictor}",
            )
        steps.append(convertor(gp_model, predictor, input_vars, output_vars, **kwargs))
        if self._output is None:
            self._output = steps[-1].output

    def print_stats(self, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        The pipeline version includes a summary of the steps that it contains.

        Parameters
        ----------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(file=file)
        print(file=file)
        print(f"Pipeline has {len(self._steps)} steps:", file=file)
        print(file=file)

        self._print_container_steps("Step", self._steps, file=file)

    @property
    def _has_solution(self):
        return self[-1]._has_solution

    @property
    def output(self):
        """Returns output variables of pipeline, i.e. output of its last step."""
        return self[-1].output

    @property
    def output_values(self):
        """Returns output values of pipeline in solution, i.e. output of its last step."""
        return self[-1].output_values

    @property
    def input(self):
        """Returns input variables of pipeline, i.e. input of its first step."""
        return self[0].input

    @property
    def input_values(self):
        """Returns input values of pipeline in solution, i.e. input of its first step."""
        return self[0].input_values

    def __getitem__(self, key):
        """Get an item from the pipeline steps."""
        return self._steps[key]

    def __iter__(self):
        """Iterate through pipeline steps."""
        return self._steps.__iter__()

    def __len__(self):
        """Get number of pipeline steps."""
        return self._steps.__len__()
