# Copyright © 2022 Gurobi Optimization, LLC
''' Model decision trees based regressor from scikit learn

   Implements the decision tree and gradient boosting trees'''

import numpy as np
from gurobipy import GRB, quicksum

from .utils import MLSubModel


class DecisionTree2Gurobi(MLSubModel):
    ''' Class to model a trained decision tree in a Gurobi model'''
    def __init__(self, model, regressor, input_vars, output_vars, **kwargs):
        self.tree = regressor.tree_
        super().__init__(model, input_vars, output_vars, **kwargs)

    def mip_model(self):
        tree = self.tree
        model = self.model

        _input = self._input
        output = self._output
        nex = _input.shape[0]
        nodes = model.addMVar((nex, tree.capacity), vtype=GRB.BINARY)

        # Intermediate nodes constraints
        # Can be added all at once
        notleafs = [i for i, j in enumerate(tree.children_left) if j >= 0]
        model.addConstrs(nodes[:, i] >= nodes[:, tree.children_left[i]] for i in notleafs)
        model.addConstrs(nodes[:, i] >= nodes[:, tree.children_right[i]] for i in notleafs)
        model.addConstrs(nodes[:, i] <= nodes[:, tree.children_right[i]] +
                     nodes[:, tree.children_left[i]] for i in notleafs)
        model.addConstrs(nodes[:, tree.children_right[i]] +
                     nodes[:, tree.children_left[i]] <= 1 for i in notleafs)

        # Node splitting
        for node in range(tree.capacity):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left < 0:
                model.addConstrs((nodes[k, node] == 1) >> (output[k, 0] == tree.value[node][0][0])
                             for k in range(nex))
                continue
            model.addConstrs((nodes[k, left] == 1) >>
                         (_input[k, tree.feature[node]] <= tree.threshold[node])
                         for k in range(nex))
            model.addConstrs((nodes[k, right] == 1) >>
                         (_input[k, tree.feature[node]] >= tree.threshold[node] + 1e-8)
                         for k in range(nex))

        # We should attain 1 leaf
        model.addConstrs(quicksum([nodes[k, i] for i in range(tree.capacity)
                              if tree.children_left[i] < 0]) == 1
                     for k in range(_input.shape[0]))

        output.LB = np.min(tree.value)
        output.UB = np.max(tree.value)


class GradientBoostingRegressor2Gurobi(MLSubModel):
    ''' Class to model a trained gradient boosting tree in a Gurobi model'''
    def __init__(self, model, regressor, input_vars, output_vars):
        self.regressor = regressor
        super().__init__(model, input_vars, output_vars)

    def mip_model(self):
        ''' Predict output variables y from input variables X using the
            decision tree.

            Both X and y should be array or list of variables of conforming dimensions.
        '''
        model = self.model
        regressor = self.regressor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        treevars = model.addMVar((nex, regressor.n_estimators_), lb=-GRB.INFINITY)
        constant = regressor.init_.constant_

        tree2gurobi = []
        for i in range(regressor.n_estimators_):
            tree = regressor.estimators_[i]
            tree2gurobi.append(DecisionTree2Gurobi(model, tree[0], _input,treevars[:, i])._add())
        for k in range(nex):
            model.addConstr(output[k, :] == regressor.learning_rate * treevars[k, :].sum()
                            + constant)
