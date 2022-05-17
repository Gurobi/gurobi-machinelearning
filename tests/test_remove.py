import random
import unittest
import warnings

import gurobipy as gp
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml2gurobi.sklearn import (
    DecisionTree2Gurobi,
    GradientBoostingRegressor2Gurobi,
    LinearRegression2Gurobi,
    MLPRegressor2Gurobi,
    Pipe2Gurobi,
)


class TestFormulations(unittest.TestCase):

    def add_remove(self, regressor, translator, X, y, exampleno):
        with gp.Model() as m:
            x = m.addMVar(X.shape[1], lb=X[exampleno,:], ub=X[exampleno,:])
            y = m.addMVar(1, lb=-gp.GRB.INFINITY)
            m.update()
            numVars = m.NumVars

            m.Params.OutputFlag = 0
            reg2gurobi = translator(regressor, model=m)
            assert m.NumVars == numVars + reg2gurobi.NumVars
            assert m.NumConstrs == reg2gurobi.NumConstrs
            assert m.NumQConstrs == reg2gurobi.NumQConstrs
            assert m.NumGenConstrs == reg2gurobi.NumGenConstrs
            reg2gurobi.predict(x, y)
            m.update()
            assert m.NumVars == numVars + reg2gurobi.NumVars
            assert m.NumConstrs == reg2gurobi.NumConstrs
            assert m.NumQConstrs == reg2gurobi.NumQConstrs
            assert m.NumGenConstrs == reg2gurobi.NumGenConstrs
            reg2gurobi.remove()
            assert m.NumVars == numVars
            assert m.NumConstrs == 0
            assert m.NumGenConstrs == 0
            assert m.NumQConstrs == 0

    def test_diabetes(self):
        data = datasets.load_diabetes()

        X = data['data']
        y = data['target']

        to_test = [(LinearRegression(), LinearRegression2Gurobi),
                   (DecisionTreeRegressor(max_leaf_nodes=50), DecisionTree2Gurobi),
                   (GradientBoostingRegressor(n_estimators=20), GradientBoostingRegressor2Gurobi),
                   (MLPRegressor([20,20]), MLPRegressor2Gurobi)]

        warnings.filterwarnings('ignore')
        for regressor, translator in to_test:
            regressor.fit(X, y)
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, translator=translator, exampleno=exampleno):
                    self.add_remove(regressor, translator, X, y, exampleno)

        return
        for regressor, _ in to_test:
            pipeline = make_pipeline(StandardScaler(), regressor)
            pipeline.fit(X, y)
            for _ in range(5):
                exampleno=random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, translator=translator, exampleno=exampleno):
                    self.add_remove(pipeline, Pipe2Gurobi, X, y, exampleno)
