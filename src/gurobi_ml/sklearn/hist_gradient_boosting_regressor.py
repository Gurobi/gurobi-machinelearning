# Copyright © 2023-2026 Gurobi Optimization, LLC
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
:external+sklearn:py:class:`sklearn.ensemble.HistGradientBoostingRegressor`
into a :external+gurobi:py:class:`Model`.
"""

import numpy as np
from gurobipy import GRB

from ..exceptions import ModelConfigurationError
from ..modeling import AbstractPredictorConstr
from ..modeling.decision_tree import AbstractTreeEstimator
from .skgetter import SKgetter


def add_hist_gradient_boosting_regressor_constr(
    gp_model,
    hist_gradient_boosting_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    safety_floor=0.0,
    **kwargs,
):
    """Formulate hist_gradient_boosting_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to hist_gradient_boosting_regressor.
    """
    return HistGradientBoostingRegressorConstr(
        gp_model,
        hist_gradient_boosting_regressor,
        input_vars,
        output_vars,
        epsilon=epsilon,
        safety_floor=safety_floor,
        **kwargs,
    )


class HistGradientBoostingRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained HistGradientBoostingRegressor."""

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        epsilon=0.0,
        safety_floor=0.0,
        **kwargs,
    ):
        self._output_shape = 1
        self._default_name = "hist_gbtree_reg"
        self.epsilon = epsilon
        self.safety_floor = safety_floor
        self.estimators_ = []
        SKgetter.__init__(self, predictor, input_vars)
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        predictor = self.predictor
        model = self.gp_model
        output = self._output
        _input = self._input
        n_examples = _input.shape[0]

        if output.shape[1] != 1:
            raise ModelConfigurationError(
                predictor, "Output dimension of gradient boosting regressor should be 1"
            )

        if predictor.loss not in ("squared_error", "absolute_error", "quantile"):
            raise ModelConfigurationError(
                predictor,
                f"loss '{predictor.loss}' is not supported; only identity-link losses "
                "are supported",
            )

        if self._no_debug:
            kwargs["no_record"] = True

        trees = [tree for iteration in predictor._predictors for tree in iteration]
        tree_vars = model.addMVar(
            (n_examples, len(trees), 1),
            lb=-GRB.INFINITY,
            name=self._name_var("estimator"),
        )

        estimators = []
        timer = AbstractPredictorConstr._ModelingTimer()
        for i, tree in enumerate(trees):
            nodes = tree.nodes
            if "is_categorical" in nodes.dtype.names and nodes["is_categorical"].any():
                raise ModelConfigurationError(
                    predictor, "categorical splits are not supported"
                )

            children_left = nodes["left"].astype(np.intp)
            children_right = nodes["right"].astype(np.intp)
            is_leaf = nodes["is_leaf"].astype(bool)
            children_left[is_leaf] = -1
            children_right[is_leaf] = -1
            tree_dict = {
                "children_left": children_left,
                "children_right": children_right,
                "feature": nodes["feature_idx"],
                "threshold": nodes["num_threshold"],
                "value": nodes["value"].reshape(-1, 1),
                "capacity": len(nodes),
                "n_features": predictor.n_features_in_,
            }
            estimators.append(
                AbstractTreeEstimator(
                    model,
                    tree_dict,
                    self.input,
                    tree_vars[:, i, :],
                    self.epsilon,
                    timer,
                    safety_floor=self.safety_floor,
                    **kwargs,
                )
            )
        self.estimators_ = estimators

        baseline = predictor._baseline_prediction[0, 0]
        model.addConstr(output == tree_vars.sum(axis=1) + baseline)
