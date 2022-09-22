# Copyright © 2022 Gurobi Optimization, LLC
""" Model decision trees based regressor from scikit learn

   Implements the decision tree and gradient boosting trees"""

import numpy as np
from gurobipy import GRB

from ..basepredictor import AbstractPredictorConstr


class DecisionTreeRegressorConstr(AbstractPredictorConstr):
    """Class to model a trained decision tree in a Gurobi model"""

    def __init__(self, grbmodel, regressor, input_vars, output_vars, **kwargs):
        self.tree = regressor.tree_
        self.n_outputs_ = regressor.n_outputs_
        super().__init__(grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
        tree = self.tree
        model = self._model

        _input = self._input
        output = self._output
        nex = _input.shape[0]
        nodes = model.addMVar((nex, tree.capacity), vtype=GRB.BINARY, name="node")
        self.nodevars = nodes

        # Intermediate nodes constraints
        # Can be added all at once
        notleafs = tree.children_left >= 0
        leafs = tree.children_left < 0
        model.addConstr(nodes[:, notleafs] >= nodes[:, tree.children_left[notleafs]])
        model.addConstr(nodes[:, notleafs] >= nodes[:, tree.children_right[notleafs]])
        model.addConstr(nodes[:, notleafs] <= nodes[:, tree.children_right[notleafs]] + nodes[:, tree.children_left[notleafs]])
        model.addConstr(nodes[:, tree.children_right[notleafs]] + nodes[:, tree.children_left[notleafs]] <= 1)

        # Node splitting
        for node in range(tree.capacity):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left >= 0:
                model.addConstrs(
                    (nodes[k, left].item() == 1) >> (_input[k, tree.feature[node]] <= tree.threshold[node]) for k in range(nex)
                )
                model.addConstrs(
                    (nodes[k, right].item() == 1) >> (_input[k, tree.feature[node]] >= tree.threshold[node] + 1e-6)
                    for k in range(nex)
                )
            else:
                model.addConstrs((nodes[k, node].item() == 1) >> (output[k, 0] == tree.value[node][0][0]) for k in range(nex))

        # We should attain 1 leaf
        model.addConstr(nodes[:, leafs].sum(axis=1) == 1)

        output.LB = np.min(tree.value)
        output.UB = np.max(tree.value)


class GradientBoostingRegressorConstr(AbstractPredictorConstr):
    """Class to model a trained gradient boosting tree in a Gurobi model"""

    def __init__(self, model, regressor, input_vars, output_vars):
        self.regressor = regressor
        self.n_outputs_ = 1
        super().__init__(model, input_vars, output_vars)

    def mip_model(self):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._model
        regressor = self.regressor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        treevars = model.addMVar((nex, regressor.n_estimators_), lb=-GRB.INFINITY)
        constant = regressor.init_.constant_

        tree2gurobi = []
        for i in range(regressor.n_estimators_):
            tree = regressor.estimators_[i]
            tree2gurobi.append(DecisionTreeRegressorConstr(model, tree[0], _input, treevars[:, i]))
        model.addConstr(output == regressor.learning_rate * treevars.sum(axis=1) + constant)


class RandomForestRegressorConstr(AbstractPredictorConstr):
    """Class to model a trained random forest regressor in a Gurobi model"""

    def __init__(self, model, regressor, input_vars, output_vars):
        self.regressor = regressor
        self.n_outputs_ = regressor.n_outputs_
        super().__init__(model, input_vars, output_vars)

    def mip_model(self):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._model
        regressor = self.regressor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        treevars = model.addMVar((nex, regressor.n_estimators), lb=-GRB.INFINITY)

        tree2gurobi = []
        for i in range(regressor.n_estimators):
            tree = regressor.estimators_[i]
            tree2gurobi.append(DecisionTreeRegressorConstr(model, tree, _input, treevars[:, i]))
        for k in range(nex):
            model.addConstr(regressor.n_estimators * output[k, :] == treevars[k, :].sum())
