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

"""Module for formulating a
XGBoost gradient boosting regressor
into a :gurobipy:`model`.
"""


import json

import numpy as np
import xgboost as xgb
from gurobipy import GRB

from ..exceptions import NoSolution
from ..modeling import AbstractPredictorConstr
from ..modeling.decision_tree import leaf_formulation


def add_xgboost_regressor_constr(
    gp_model, xgboost_regressor, input_vars, output_vars=None, epsilon=0.0, **kwargs
):
    """Formulate gradient_boosting_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to gradient_boosting_regressor. See our :ref:`User's Guide
    <Gradient Boosting Regression>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    xgboost_regressor : :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for gradient boosting regressor in model.

    Returns
    -------
    GradientBoostingRegressorConstr
        Object containing information about what was added to gp_model to formulate
        gradient_boosting_regressor.

    Note
    ----
    |VariablesDimensionsWarn|

    Also see
    :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor`
    for specific parameters to model decision tree estimators.
    """
    return XGBoostRegressorConstr(
        gp_model, xgboost_regressor, input_vars, output_vars, epsilon=epsilon, **kwargs
    )


class XGBoostRegressorConstr(AbstractPredictorConstr):
    """Class to model trained
    :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
    with gurobipy.

    |ClassShort|
    """

    def __init__(
        self, gp_model, xgb_regressor, input_vars, output_vars, epsilon=0.0, **kwargs
    ):
        self._output_shape = 1
        self.estimators_ = []
        self.xgb_regressor = xgb_regressor
        self._default_name = "xgb_reg"
        self.epsilon = epsilon
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._gp_model
        xgb_regressor = self.xgb_regressor

        _input = self._input
        output = self._output
        nex = _input.shape[0]
        timer = AbstractPredictorConstr._ModelingTimer()
        outdim = output.shape[1]
        assert (
            outdim == 1
        ), "Output dimension of gradient boosting regressor should be 1"

        xgb_raw = json.loads(xgb_regressor.save_raw(raw_format="json"))
        trees = xgb_raw["learner"]["gradient_booster"]["model"]["trees"]
        n_estimators = len(trees)

        estimators = []
        if self._no_debug:
            kwargs["no_record"] = True

        tree_vars = model.addMVar(
            (nex, n_estimators, 1),
            lb=-GRB.INFINITY,
            name=self._name_var("esimator"),
        )

        for i, tree in enumerate(trees):
            if self.verbose:
                self._timer.timing(f"Estimator {i}")
            tree["threshold"] = (
                np.array(tree["split_conditions"], dtype=np.float32) - self.epsilon
            )
            tree["children_left"] = np.array(tree["left_children"])
            tree["children_right"] = np.array(tree["right_children"])
            tree["feature"] = np.array(tree["split_indices"])
            tree["value"] = tree["threshold"].reshape(-1, 1)
            tree["capacity"] = len(tree["split_conditions"])
            tree["n_features"] = int(tree["tree_param"]["num_feature"])
            estimators.append(
                leaf_formulation(
                    self.gp_model,
                    self.input,
                    tree_vars[:, i, :],
                    tree,
                    self.epsilon,
                    self._name_var,
                    self.verbose,
                    timer,
                )
            )

        self.estimators_ = estimators

        constant = float(xgb_raw["learner"]["learner_model_param"]["base_score"])
        learning_rate = 1.0
        model.addConstr(output == learning_rate * tree_vars.sum(axis=1) + constant)

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        Includes a summary of the estimators that it contains.

        Arguments
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(abbrev=abbrev, file=file)
        if abbrev or self._no_debug:
            return
        print(file=file)

        # self._print_container_steps("Estimator", self.estimators_, file=file)

    def get_error(self, verbose=False):
        if self._has_solution:
            xgb_in = xgb.DMatrix(self.input_values)
            xgb_out = self.xgb_regressor.predict(xgb_in)
            r_val = np.abs(xgb_out.reshape(-1, 1) - self.output.X)
            if verbose:
                print(f"{self.output.X} != {xgb_out.reshape(-1, 1)}")
            return np.abs(xgb_out.reshape(-1, 1) - self.output.X)
        raise NoSolution()
