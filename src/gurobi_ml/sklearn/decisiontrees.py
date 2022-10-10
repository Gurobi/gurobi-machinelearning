# Copyright © 2022 Gurobi Optimization, LLC
""" Model decision trees based regressor from scikit learn

   Implements the decision tree and gradient boosting trees"""

import numpy as np
from gurobipy import GRB

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


class DecisionTreeRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model a trained decision tree in a Gurobi model"""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        self.n_outputs_ = predictor.n_outputs_
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
        tree = self.predictor.tree_
        model = self._model

        _input = self._input
        output = self._output
        outdim = output.shape[1]
        if outdim != 1:
            raise Exception("Can only deal with 1-dimensional regression. Output dimension {}".format(outdim))
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


class GradientBoostingRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model a trained gradient boosting tree in a Gurobi model"""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        self.n_outputs_ = 1
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._model
        predictor = self.predictor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        outdim = output.shape[1]
        if outdim != 1:
            raise Exception("Can only deal with 1-dimensional regression. Output dimension {}".format(outdim))
        treevars = model.addMVar((nex, predictor.n_estimators_), lb=-GRB.INFINITY, name="estimator")
        constant = predictor.init_.constant_

        tree2gurobi = []
        for i in range(predictor.n_estimators_):
            tree = predictor.estimators_[i]
            tree2gurobi.append(DecisionTreeRegressorConstr(model, tree[0], _input, treevars[:, i], default_name="gbt_tree"))

        model.addConstr(output[:, 0] == predictor.learning_rate * treevars.sum(axis=1) + constant[0][0])


class RandomForestRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model a trained random forest regressor in a Gurobi model"""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        self.n_outputs_ = predictor.n_outputs_
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._model
        predictor = self.predictor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        outdim = output.shape[1]
        if outdim != 1:
            raise Exception("Can only deal with 1-dimensional regression. Output dimension {}".format(outdim))
        treevars = model.addMVar((nex, predictor.n_estimators), lb=-GRB.INFINITY, name="estimator")

        tree2gurobi = []
        for i in range(predictor.n_estimators):
            tree = predictor.estimators_[i]
            tree2gurobi.append(DecisionTreeRegressorConstr(model, tree, _input, treevars[:, i], default_name="rf_tree"))

        model.addConstr(predictor.n_estimators * output[:, 0] == treevars.sum(axis=1))


def add_decision_tree_regressor_constr(grbmodel, decision_tree_regressor, input_vars, output_vars=None, **kwargs):
    return DecisionTreeRegressorConstr(grbmodel, decision_tree_regressor, input_vars, output_vars, **kwargs)


def add_gradient_boosting_regressor_constr(grbmodel, gradient_boosting_regressor, input_vars, output_vars=None, **kwargs):
    return DecisionTreeRegressorConstr(grbmodel, gradient_boosting_regressor, input_vars, output_vars, **kwargs)


def add_random_forest_regressor_constr(grbmodel, random_forest_regressor, input_vars, output_vars=None, **kwargs):
    return DecisionTreeRegressorConstr(grbmodel, random_forest_regressor, input_vars, output_vars, **kwargs)
