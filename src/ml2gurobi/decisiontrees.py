# Copyright © 2022 Gurobi Optimization, LLC
import numpy as np
from gurobipy import GRB, quicksum

from .utils import Submodel


class DecisionTree2Gurobi(Submodel):
    ''' Class to model a trained decision tree in a Gurobi model'''
    def __init__(self, regressor, model):
        super().__init__(model)
        self.tree = regressor.tree_

    def mip_model(self, X, y):
        tree = self.tree
        m = self.model

        nodes = m.addMVar((X.shape[0], tree.capacity), vtype=GRB.BINARY)

        # Intermediate nodes constraints
        # Can be added all at once
        notleafs = [i for i, j in enumerate(tree.children_left) if j >= 0]
        m.addConstrs(nodes[:, i] >= nodes[:, tree.children_left[i]] for i in notleafs)
        m.addConstrs(nodes[:, i] >= nodes[:, tree.children_right[i]] for i in notleafs)
        m.addConstrs(nodes[:, i] <= nodes[:, tree.children_right[i]] +
                     nodes[:, tree.children_left[i]] for i in notleafs)
        m.addConstrs(nodes[:, tree.children_right[i]] +
                     nodes[:, tree.children_left[i]] <= 1 for i in notleafs)

        # Node splitting
        for node in range(tree.capacity):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left < 0:
                m.addConstrs((nodes[k, node] == 1) >> (y[k, 0] == tree.value[node][0][0])
                             for k in range(X.shape[0]))
                continue
            m.addConstrs((nodes[k, left] == 1) >>
                         (X[k, tree.feature[node]] <= tree.threshold[node])
                         for k in range(X.shape[0]))
            m.addConstrs((nodes[k, right] == 1) >>
                         (X[k, tree.feature[node]] >= tree.threshold[node] + 1e-8)
                         for k in range(X.shape[0]))

        # We should attain 1 leaf
        m.addConstrs(quicksum([nodes[k, i] for i in range(tree.capacity)
                              if tree.children_left[i] < 0]) == 1
                     for k in range(X.shape[0]))

        y.LB = np.min(tree.value)
        y.UB = np.max(tree.value)


class GradientBoostingRegressor2Gurobi(Submodel):
    def __init__(self, regressor, model):
        super().__init__(model)
        self.regressor = regressor


    def mip_model(self, X, y):
        ''' Predict output variables y from input variables X using the
            decision tree.

            Both X and y should be array or list of variables of conforming dimensions.
        '''
        m = self.model
        regressor = self.regressor

        treevars = m.addMVar((X.shape[0], regressor.n_estimators_), lb = -GRB.INFINITY)
        constant = regressor.init_.constant_

        tree2gurobi = []
        for i in range(regressor.n_estimators_):
            tree = regressor.estimators_[i]
            tree2gurobi.append(DecisionTree2Gurobi(tree[0], m))
            tree2gurobi[-1].predict(X, treevars[:, i])
        for k in range(X.shape[0]):
            m.addConstr(y[k, :] == regressor.learning_rate * treevars[k,:].sum() + constant)
