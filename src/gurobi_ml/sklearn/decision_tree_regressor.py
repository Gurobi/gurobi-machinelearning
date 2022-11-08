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

""" Module for embedding a :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
into a :gurobipy:`model`.
"""

import numpy as np
from gurobipy import GRB

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def add_decision_tree_regressor_constr(
    gp_model,
    decision_tree_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    scale=1.0,
    float_type=np.float32,
    **kwargs
):
    """Embed decision_tree_regressor into gp_model

    Predict the values of output_vars using input_vars

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    decision_tree_regressor: :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
        The decision tree regressor to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for decision tree in model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for decision tree in model.
    epsilon: float, optional
        Small value used to impose strict inequalities for splitting nodes in MIP formulations.
    scale: float, optional
        Value
    float_type: type, optional
        Float type for the thresholds defining the node splits in the MIP formulation

    Returns
    -------
    DecisionTreeRegressorConstr
        Object containing information about what was added to gp_model to embed the
        predictor into it


    Note
    ----
    |VariablesDimensionsWarn|

    Warning
    -------
    Although decision trees with multiple outputs are tested they were never used in
    a non-trivial optimization model. It should be used with care at this point.
    """
    return DecisionTreeRegressorConstr(
        gp_model,
        decision_tree_regressor,
        input_vars,
        output_vars,
        epsilon,
        scale,
        float_type,
        **kwargs
    )


class DecisionTreeRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor` with gurobipy

    Stores the changes to :gurobipy:`model` when embedding an instance into it."""

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        epsilon=0.0,
        scale=1.0,
        float_type=np.float32,
        **kwargs
    ):
        self.n_outputs_ = predictor.n_outputs_
        self.epsilon = epsilon
        self.scale = scale
        self.float_type = float_type
        self._default_name = "tree_reg"
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, gp_model, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        tree = self.predictor.tree_
        model = self._gp_model

        _input = self._input
        output = self._output
        outdim = output.shape[1]
        assert outdim == self.n_outputs_
        nex = _input.shape[0]
        nodes = model.addMVar((nex, tree.capacity), vtype=GRB.BINARY, name="node")
        self.nodevars = nodes

        # Intermediate nodes constraints
        # Can be added all at once
        notleafs = tree.children_left >= 0
        leafs = tree.children_left < 0
        model.addConstr(nodes[:, notleafs] >= nodes[:, tree.children_left[notleafs]])
        model.addConstr(nodes[:, notleafs] >= nodes[:, tree.children_right[notleafs]])
        model.addConstr(
            nodes[:, notleafs]
            <= nodes[:, tree.children_right[notleafs]] + nodes[:, tree.children_left[notleafs]]
        )
        model.addConstr(
            nodes[:, tree.children_right[notleafs]] + nodes[:, tree.children_left[notleafs]] <= 1
        )

        # Node splitting
        for node in range(tree.capacity):
            left = tree.children_left[node]
            right = tree.children_right[node]
            threshold = tree.threshold[node]
            threshold = self.float_type(threshold)
            scale = max(abs(1 / threshold), self.scale)
            if left >= 0:
                # Intermediate node
                model.addConstrs(
                    (nodes[k, left].item() == 1)
                    >> (scale * _input[k, tree.feature[node]] <= scale * threshold)
                    for k in range(nex)
                )
                model.addConstrs(
                    (nodes[k, right].item() == 1)
                    >> (scale * _input[k, tree.feature[node]] >= scale * threshold + self.epsilon)
                    for k in range(nex)
                )
            else:
                # Leaf node:
                model.addConstrs(
                    (nodes[k, node].item() == 1) >> (output[k, i] == tree.value[node][i][0])
                    for i in range(self.n_outputs_)
                    for k in range(nex)
                )

        # We should attain 1 leaf
        model.addConstr(nodes[:, leafs].sum(axis=1) == 1)

        output.LB = np.min(tree.value)
        output.UB = np.max(tree.value)
