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
LightGBM gradient boosting regressor
into a :external+gurobi:py:class:`Model`.
"""

import numpy as np
from gurobipy import GRB

from ..exceptions import NoSolutionError
from ..modeling import AbstractPredictorConstr
from ..modeling.decision_tree import AbstractTreeEstimator


def add_lgbmregressor_constr(
    gp_model, lgbm_regressor, input_vars, output_vars=None, epsilon=0.0, **kwargs
):
    """Formulate lgbm_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to lgbm_regressor. See our :ref:`User's Guide
    <Gradient Boosting Regression>` for details on the mip formulation used.

    This version is for using directly with the Scikit-Learn wrapper of lgbm.
    Note that only "gbtree" regressors are supported at this point.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    lgbm_regressor : :external+lightgbm:py:class:`lightgbm.sklearn.LGBMRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for gradient boosting regressor in model.

    Returns
    -------
    LightGBMRegressorConstr
        Object containing information about what was added to gp_model to formulate
        gradient_boosting_regressor.

    Notes
    -----
    |VariablesDimensionsWarn|

    Also see
    :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor`
    for specific parameters to model decision tree estimators.

    Raises
    ------
    NoModel
        If the booster is not of type "gbtree".
    """
    return LGBMConstr(
        gp_model,
        lgbm_regressor.booster_,
        input_vars,
        output_vars,
        epsilon=epsilon,
        **kwargs,
    )


def add_lgbm_booster_constr(
    gp_model, lgbm_booster, input_vars, output_vars=None, epsilon=0.0, **kwargs
):
    """Formulate lgbm_booster into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to lgbm_regressor. See our :ref:`User's Guide
    <Gradient Boosting Regression>` for details on the mip formulation used.

    Note that only "gbtree" regressors are supported at this point.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    lgbm_regressor : :external+lightgbm:py:class:`lightgbm.Booster`
        The booster to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for gradient boosting regressor in model.

    Returns
    -------
    LightGBMRegressorConstr
        Object containing information about what was added to gp_model to formulate
        gradient_boosting_regressor.

    Notes
    -----
    |VariablesDimensionsWarn|

    Also see
    :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor`
    for specific parameters to model decision tree estimators.

    Raises
    ------
    NoModel
        If the booster is not of type "gbtree".
    """
    return LGBMConstr(
        gp_model, lgbm_booster, input_vars, output_vars, epsilon=epsilon, **kwargs
    )


class LGBMConstr(AbstractPredictorConstr):
    """Class to model trained :external+lightgbm:py:class:`lightgbm.Booster`
    in a gurobipy model.

    |ClassShort|
    """

    def __init__(
        self, gp_model, lgbm_regressor, input_vars, output_vars, epsilon=0.0, **kwargs
    ):
        self._output_shape = 1
        self.estimators_ = []
        self.lgbm_regressor = lgbm_regressor
        self._default_name = "lgbm_reg"
        self.epsilon = epsilon
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    @staticmethod
    def _count_nodes(root_node):
        """Count the nodes in a lightgbm tree.

        Traverse the tree and count the number of leaf and split nodes.

        Returns
        -------
        num_leafs : int
            Number of leaf nodes in the tree.
        num_split : int
            Number of split nodes in the tree.
        """
        heap = [root_node]
        num_leafs = 0
        num_split = 0
        while len(heap) > 0:
            node = heap.pop()
            if "split_index" in node:
                heap.append(node["left_child"])
                heap.append(node["right_child"])
                num_split += 1
            else:
                num_leafs += 1
        return (num_leafs, num_split)

    @staticmethod
    def _flat_tree_representation(root_node):
        """Flatten a lightgbm tree.

        This function takes a root node of a lightgbm tree and flattens it into a dictionary representation.
        The flattened tree contains information about the children nodes, split features, thresholds, and leaf values.

        Args:
            root_node (dict): The root node of the lightgbm tree.

        Returns:
            dict: A dictionary representation of the flattened tree.

        """
        num_leafs, num_split = LGBMConstr._count_nodes(root_node)

        numnodes = num_leafs + num_split
        children_left = np.full(numnodes, -2, dtype=int)
        children_right = np.full(numnodes, -2, dtype=int)
        feature = np.empty(numnodes, dtype=int)
        value = np.empty(numnodes, dtype=np.float32)
        threshold = np.empty(numnodes, dtype=np.float32)

        def node_index(node):
            if "split_index" in node.keys():
                return node["split_index"]
            return node["leaf_index"] + num_split

        heap = [root_node]
        while len(heap) > 0:
            node = heap.pop()
            index = node_index(node)
            assert children_left[index] == -2
            assert children_right[index] == -2
            if "split_index" in node.keys():
                feature[index] = node["split_feature"]
                threshold[index] = node["threshold"]
                children_left[index] = node_index(node["left_child"])
                children_right[index] = node_index(node["right_child"])

                heap.append(node["left_child"])
                heap.append(node["right_child"])
            else:
                children_left[index] = -1
                children_right[index] = -1
                value[index] = node["leaf_value"]

        assert min(children_left) >= -1
        assert min(children_right) >= -1
        flat_tree = {
            "children_left": children_left,
            "children_right": children_right,
            "feature": feature,
            "threshold": threshold,
            "value": value.reshape(-1, 1),
            "capacity": numnodes,
        }
        return flat_tree

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self.gp_model
        lgbm_regressor = self.lgbm_regressor

        _input = self._input
        output = self._output
        nex = _input.shape[0]
        timer = AbstractPredictorConstr._ModelingTimer()
        outdim = output.shape[1]
        assert outdim == 1, (
            "Output dimension of gradient boosting regressor should be 1"
        )

        lgbm_raw = lgbm_regressor.dump_model()

        trees = lgbm_raw["tree_info"]
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
            flat_tree = self._flat_tree_representation(tree["tree_structure"])
            flat_tree["n_features"] = lgbm_raw["max_feature_idx"] + 1

            def _name_tree_var(name):
                rval = self._name_var(name)
                if rval is None:
                    return None
                return rval + f"_{i}"

            estimators.append(
                AbstractTreeEstimator(
                    self.gp_model,
                    flat_tree,
                    self.input,
                    tree_vars[:, i, :],
                    self.epsilon,
                    timer,
                    **kwargs,
                )
            )

        self.estimators_ = estimators

        model.addConstr(output == tree_vars.sum(axis=1))

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

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
            lgbm_in = self.input_values
            lgbm_out = self.lgbm_regressor.predict(lgbm_in)
            r_val = np.abs(lgbm_out.reshape(-1, 1) - self.output.X)
            if eps is not None and np.max(r_val) > eps:
                print(f"{self.output.X} != {lgbm_out.reshape(-1, 1)}")
            return r_val
        raise NoSolutionError()
