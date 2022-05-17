import random
import unittest

import gurobipy as gp
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml2grb import sklearn2grb


class TestFormulations(unittest.TestCase):

    def fixed_model(self, regressor, translator, X, y, exampleno):
        with gp.Model() as m:
            x = m.addMVar(X.shape[1], lb=X[exampleno,:], ub=X[exampleno,:])
            y = m.addMVar(1, lb=-gp.GRB.INFINITY)

            y.shape

            reg2grb = translator(regressor, model=m)
            reg2grb.predict(x, y)
            m.Params.OutputFlag = 0
            m.optimize()

            self.assertTrue(abs(y.X - regressor.predict(X[exampleno,:].reshape(1, -1))) < 1e-5)

    def test_diabetes(self):
        data = datasets.load_diabetes()

        X = data['data']
        y = data['target']

        to_test = [(LinearRegression(), sklearn2grb.LinearRegression2Grb),
                   (DecisionTreeRegressor(max_leaf_nodes=50), sklearn2grb.DecisionTree2Grb),
                   (GradientBoostingRegressor(n_estimators=20), sklearn2grb.GradientBoostingRegressor2Gurobi),
                   (MLPRegressor([20,20]), sklearn2grb.MLPRegressor2Grb)]

        for regressor, translator in to_test:
            regressor.fit(X, y)
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, translator=translator, exampleno=exampleno):
                    self.fixed_model(regressor, translator, X, y, exampleno)

        for regressor, _ in to_test:
            pipeline = make_pipeline(StandardScaler(), regressor)
            pipeline.fit(X, y)
            for _ in range(5):
                exampleno=random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, translator=translator, exampleno=exampleno):
                    self.fixed_model(pipeline, sklearn2grb.Pipe2Gurobi, X, y, exampleno)