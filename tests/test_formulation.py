import random
import unittest
import warnings

import gurobipy as gp
import numpy as np
from joblib import load
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml2gurobi.activations import Identity
from ml2gurobi.extra import morerelu
from ml2gurobi.extra.obbt import obbt
from ml2gurobi.sklearn import (
    DecisionTree2Gurobi,
    GradientBoostingRegressor2Gurobi,
    LinearRegression2Gurobi,
    MLPRegressor2Gurobi,
    Pipe2Gurobi,
)

gp.gurobi._nd_unleashed = True


class TestFormulations(unittest.TestCase):
    def fixed_model(self, regressor, translator, X, y, exampleno):
        with gp.Model() as m:
            x = m.addMVar(X.shape[1], lb=X[exampleno, :], ub=X[exampleno, :])
            y = m.addMVar(1, lb=-gp.GRB.INFINITY)

            y.shape

            m.Params.OutputFlag = 0
            reg2gurobi = translator(regressor, model=m)
            reg2gurobi.predict(x, y)
            m.optimize()

            self.assertTrue(abs(y.X - regressor.predict(X[exampleno, :].reshape(1, -1))) < 1e-5)

    def test_diabetes(self):
        data = datasets.load_diabetes()

        X = data['data']
        y = data['target']

        to_test = [(LinearRegression(), LinearRegression2Gurobi),
                   (DecisionTreeRegressor(max_leaf_nodes=50), DecisionTree2Gurobi),
                   (GradientBoostingRegressor(n_estimators=20), GradientBoostingRegressor2Gurobi),
                   (MLPRegressor([20, 20]), MLPRegressor2Gurobi)]

        warnings.filterwarnings('ignore')
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
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, translator=translator, exampleno=exampleno):
                    self.fixed_model(pipeline, Pipe2Gurobi, X, y, exampleno)

    def adversarial_model(self, m, pipe, example, epsilon, activation=None):
        ex_prob = pipe.predict_proba(example)
        output_shape = ex_prob.shape
        sortedidx = np.argsort(ex_prob)[0]

        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name='X')
        absdiff = m.addMVar(example.shape, lb=0, ub=1, name='dplus')
        output = m.addMVar(output_shape, lb=-gp.GRB.INFINITY, name='y')

        m.setObjective(output[0, sortedidx[-2]] - output[0, sortedidx[-1]],
               gp.GRB.MAXIMIZE)

        # Bound on the distance to example in norm-2
        m.addConstr(absdiff[0, :] >= x[0, :] - example.to_numpy()[0, :])
        m.addConstr(absdiff[0, :] >= - x[0, :] + example.to_numpy()[0, :])
        m.addConstr(absdiff[0, :].sum() <= epsilon)

        # Code to add the neural network to the constraints
        pipe2gurobi = Pipe2Gurobi(pipe, m)
        # For this example we should model softmax in the last layer using identity
        if activation is not None:
            pipe2gurobi.steps[-1].actdict['relu'] = activation
        pipe2gurobi.steps[-1].actdict['softmax'] = Identity()
        pipe2gurobi.predict(x, output)
        return pipe2gurobi

    def test_adversarial_activations(self):
        # Load the trained network and the examples
        pipe = load('networks/MNIST_50_50.joblib')
        X = load('networks/MNIST_first100.joblib')

        # Choose an example
        exampleno = random.randint(0, 100)
        example = X.iloc[exampleno:exampleno+1, :]
        epsilon = 0.01
        activations = (None, morerelu.ReLUmin(), morerelu.GRBReLU(), morerelu.ReLUM())
        value = None
        for activation in activations:
            with gp.Model() as m:
                m.Params.OutputFlag = 0
                with self.subTest(example=exampleno, epsilon=epsilon, activation=activation, obbt=False):
                    self.adversarial_model(m, pipe, example, epsilon, activation=activation)
                    m.optimize()
                    if value is None:
                        value = m.ObjVal
                    else:
                        self.assertAlmostEqual(value, m.ObjVal, places=5)

    def test_adversarial_activations_obbt(self):
        # Load the trained network and the examples
        pipe = load('networks/MNIST_50_50.joblib')
        X = load('networks/MNIST_first100.joblib')

        # Choose an example
        exampleno = random.randint(0, 100)
        example = X.iloc[exampleno:exampleno+1, :]
        activations = (None, morerelu.GRBReLU(), morerelu.ReLUM())
        epsilon = 0.1
        value = None
        for activation in activations:
            with gp.Model() as m:
                m.Params.OutputFlag = 0
                with self.subTest(example=exampleno, epsilon=epsilon, activation=activation, obbt=True):
                    pipe2gurobi = self.adversarial_model(m, pipe, example, epsilon, activation=activation)
                    if activation is None:
                        activation = morerelu.reluOBBT('both')
                    obbt(pipe2gurobi.steps[-1], activation=activation)
                    m.optimize()
                    if value is None:
                        value = m.ObjVal
                    else:
                        self.assertAlmostEqual(value, m.ObjVal, places=5)
