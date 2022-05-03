# Copyright © 2022 Gurobi Optimization, LLC
import numpy as np
from gurobipy import GRB, quicksum

from .utils import transpose, validate_gpvars


class DecisionTree2Grb:
    ''' Class to model a trained decision tree in a Gurobi model'''
    def __init__(self, regressor, model):
        self.tree = regressor.tree_
        self.model = model

    @staticmethod
    def validate(input_vars, output_vars):
        input_vars = validate_gpvars(input_vars)
        output_vars = validate_gpvars(output_vars)
        if output_vars.shape[0] != input_vars.shape[0] and output_vars.shape[1] != input_vars.shape[0]:
            raise BaseException("Non-conforming dimension between input variable and output variable: {} != {}".
                                format(output_vars.shape[0], input_vars.shape[0]))
        elif input_vars.shape[0] != output_vars.shape[0] and output_vars.shape[1] == input_vars.shape[0]:
            output_vars = transpose(output_vars)

        return (input_vars, output_vars)

    def predict(self, X, y):
        ''' Predict output variables y from input variables X using the
            decision tree.

            Both X and y should be array or list of variables of conforming dimensions.
        '''
        X, y = self.validate(X, y)

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
                m.addConstrs((nodes[k, node] == 1) >> (y[k, 0] == tree.value[node][0][0]) for k in range(X.shape[0]))
                continue
            m.addConstrs((nodes[k, left] == 1) >> (X[k, tree.feature[node]] <= tree.threshold[node]) for k in range(X.shape[0]))
            m.addConstrs((nodes[k, right] == 1) >> (X[k, tree.feature[node]] >= tree.threshold[node] + 1e-8) for k in range(X.shape[0]))

        # We should attain 1 leaf
        m.addConstrs(quicksum([nodes[k, i] for i in range(tree.capacity)
                              if tree.children_left[i] < 0]) == 1
                     for k in range(X.shape[0]))

        y.LB = np.min(tree.value)
        y.UB = np.max(tree.value)
