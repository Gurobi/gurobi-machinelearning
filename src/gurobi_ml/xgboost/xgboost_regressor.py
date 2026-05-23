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
XGBoost gradient boosting regressor
into a :external+gurobi:py:class:`Model`.
"""

import json
import warnings

import gurobipy as gp
import numpy as np
import xgboost as xgb
from gurobipy import GRB

try:
    from gurobipy import nlfunc

    HAS_NLFUNC = True
except ImportError:
    HAS_NLFUNC = False

from ..exceptions import ModelConfigurationError, NoSolutionError
from ..modeling import AbstractPredictorConstr
from ..modeling.decision_tree import AbstractTreeEstimator


def add_xgbregressor_constr(
    gp_model,
    xgboost_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    formulation="leaf",
    safety_floor=0.0,
    **kwargs,
):
    """Formulate xgboost_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to xgboost_regressor. See our :ref:`User's Guide
    <Gradient Boosting Regression>` for details on the mip formulation used.

    This version is for using directly with the Scikit-Learn wrapper of XGBoost.
    Note that only "gbtree" regressors are supported at this point.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    xgboost_regressor : :external+xgb:py:class:`xgboost.XGBRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for gradient boosting regressor in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for gradient boosting regressor in gp_model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    safety_floor : float, optional
        Thresholds with absolute value smaller than this will be clamped
        to this value to avoid numerical issues with Gurobi's tolerance.

    Returns
    -------
    XGBoostRegressorConstr
        Object containing information about what was added to gp_model to formulate
        gradient_boosting_regressor.

    Notes
    -----
    |VariablesDimensionsWarn|

    Also see
    :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor_constr`
    for specific parameters to model decision tree estimators.

    Raises
    ------
    ModelConfigurationError
        If the booster is not of type "gbtree".
    """
    return XGBoostRegressorConstr(
        gp_model,
        xgboost_regressor.get_booster(),
        input_vars,
        output_vars,
        epsilon=epsilon,
        formulation=formulation,
        safety_floor=safety_floor,
        **kwargs,
    )


def add_xgboost_regressor_constr(
    gp_model,
    xgboost_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    formulation="leaf",
    safety_floor=0.0,
    **kwargs,
):
    """Formulate xgboost_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to xgboost_regressor. See our :ref:`User's Guide
    <Gradient Boosting Regression>` for details on the mip formulation used.

    Note that only "gbtree" regressors are supported at this point.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    xgboost_regressor : :external+xgb:py:class:`xgboost.Booster`
        The gradient boosting regressor to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for gradient boosting regressor in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for gradient boosting regressor in gp_model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    safety_floor : float, optional
        Thresholds with absolute value smaller than this will be clamped
        to this value to avoid numerical issues with Gurobi's tolerance.

    Returns
    -------
    XGBoostRegressorConstr
        Object containing information about what was added to gp_model to formulate
        gradient_boosting_regressor.

    Notes
    -----
    |VariablesDimensionsWarn|

    Also see
    :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor_constr`
    for specific parameters to model decision tree estimators.

    Raises
    ------
    ModelConfigurationError
        If the booster is not of type "gbtree".
    """
    return XGBoostRegressorConstr(
        gp_model,
        xgboost_regressor,
        input_vars,
        output_vars,
        epsilon=epsilon,
        formulation=formulation,
        safety_floor=safety_floor,
        **kwargs,
    )


class XGBoostRegressorConstr(AbstractPredictorConstr):
    """Class to model trained :external+xgb:py:class:`xgboost.Booster`
    in a gurobipy model.

    |ClassShort|
    """

    def __init__(
        self,
        gp_model,
        xgb_regressor,
        input_vars,
        output_vars,
        epsilon=0.0,
        formulation="leaf",
        safety_floor=0.0,
        **kwargs,
    ):
        """Initialize XGBoostRegressorConstr.

        Parameters
        ----------
        gp_model : :external+gurobi:py:class:`Model`
            The gurobipy model where the predictor should be inserted.
        xgb_regressor : :external+xgb:py:class:`xgboost.Booster`
            The booster to insert as predictor.
        input_vars : mvar_array_like
            Decision variables used as input for gradient boosting regressor in gp_model.
        output_vars : mvar_array_like, optional
            Decision variables used as output for gradient boosting regressor in gp_model.
        epsilon : float, optional
            Small value used to impose strict inequalities for splitting nodes in
            MIP formulations.
        """
        self._output_shape = 1
        self.estimators_ = []
        self.xgb_regressor = xgb_regressor
        self._default_name = "xgb_reg"
        self.epsilon = epsilon
        self.formulation = formulation
        self.safety_floor = safety_floor
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Build the MIP model for the XGBoost regressor.

        The formulation predicts the values of `self.output` using `self.input`
        according to the XGBoost booster.
        """
        model = self.gp_model
        xgb_regressor = self.xgb_regressor

        _input = self._input
        output = self._output
        nex = _input.shape[0]
        timer = AbstractPredictorConstr._ModelingTimer()
        outdim = output.shape[1]
        if outdim != 1:
            raise ModelConfigurationError(
                xgb_regressor,
                "Output dimension of gradient boosting regressor should be 1",
            )

        xgb_raw = json.loads(xgb_regressor.save_raw(raw_format="json"))
        booster_type = xgb_raw["learner"]["gradient_booster"]["name"]
        if booster_type != "gbtree":
            raise ModelConfigurationError(
                xgb_regressor, f"model not implemented for {booster_type}"
            )
        trees = xgb_raw["learner"]["gradient_booster"]["model"]["trees"]

        if self._no_debug:
            kwargs["no_record"] = True

        n_estimators = len(trees)

        estimators = []
        if self._no_debug:
            kwargs["no_record"] = True

        tree_vars = model.addMVar(
            (nex, n_estimators, 1),
            lb=-GRB.INFINITY,
            name=self._name_var("estimator"),
        )

        for i, tree in enumerate(trees):
            if self.verbose:
                self._timer.timing(f"Estimator {i}")
            raw_vals = np.array(tree["split_conditions"], dtype=np.float32)
            children_left = np.array(tree["left_children"])
            is_leaf = children_left < 0

            mip_thresholds = raw_vals.copy()
            mip_thresholds[~is_leaf] -= self.epsilon

            tree["threshold"] = mip_thresholds
            tree["children_left"] = children_left
            tree["children_right"] = np.array(tree["right_children"])
            tree["feature"] = np.array(tree["split_indices"])
            tree["value"] = raw_vals.reshape(-1, 1)
            tree["capacity"] = len(tree["split_conditions"])
            tree["n_features"] = int(tree["tree_param"]["num_feature"])

            estimators.append(
                AbstractTreeEstimator(
                    self.gp_model,
                    tree,
                    self.input,
                    tree_vars[:, i, :],
                    self.epsilon,
                    timer,
                    safety_floor=self.safety_floor,
                    formulation=self.formulation,
                    **kwargs,
                )
            )

        self.estimators_ = estimators

        base_score_raw = xgb_raw["learner"]["learner_model_param"]["base_score"]
        # In XGBoost 3.x, base_score is stored as a string representation of an array
        # In XGBoost 2.x, base_score is a string but not in array format
        if isinstance(base_score_raw, str):
            if base_score_raw.startswith("["):
                import ast

                constant = float(ast.literal_eval(base_score_raw)[0])
            else:
                constant = float(base_score_raw)
        else:
            constant = float(base_score_raw)
        learning_rate = 1.0
        objective = xgb_raw["learner"]["objective"]["name"]

        if objective in ("reg:logistic", "binary:logistic"):
            if gp.gurobi.version()[0] < 11:
                raise ModelConfigurationError(
                    xgb_regressor,
                    f"Option objective:{objective} only supported with Gurobi >= 11",
                )

            if HAS_NLFUNC:
                model.addConstr(
                    output == nlfunc.logistic(learning_rate * tree_vars.sum(axis=1))
                )
            else:
                affinevar = model.addMVar(output.shape, lb=-float("infinity"))
                model.addConstr(affinevar == learning_rate * tree_vars.sum(axis=1))
                for index in np.ndindex(self.output.shape):
                    self.gp_model.addGenConstrLogistic(
                        affinevar[index],
                        output[index],
                        name=self._indexed_name(index, "logistic"),
                    )
                num_gc = self.gp_model.NumGenConstrs
                self.gp_model.update()
                for gen_constr in self.gp_model.getGenConstrs()[num_gc:]:
                    gen_constr.setAttr("FuncNonLinear", 1)
        elif objective == "reg:squarederror":
            model.addConstr(output == learning_rate * tree_vars.sum(axis=1) + constant)
        else:
            raise ModelConfigurationError(
                xgb_regressor, f"objective type '{objective}' not implemented"
            )

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that were added to the model.

        Includes a summary of the estimators that it contains.

        Parameters
        ----------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(abbrev=abbrev, file=file)
        if abbrev or self._no_debug:
            return
        print(file=file)

        # self._print_container_steps("Estimator", self.estimators_, file=file)

    def get_error(self, eps=None):
        if self._has_solution:
            xgb_in = xgb.DMatrix(self.input_values)
            xgb_out = self.xgb_regressor.predict(xgb_in)
            r_val = np.abs(xgb_out.reshape(-1, 1) - self.output.X)
            if eps is not None and np.max(r_val) > eps:
                warnings.warn(f"get_error: {self.output.X} != {xgb_out.reshape(-1, 1)}")
            return r_val
        raise NoSolutionError()
