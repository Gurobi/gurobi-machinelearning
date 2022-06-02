import os
import random
import unittest
import warnings

try:
    import gurobi.modeling as gp
    HAS_MODELING=True
except ImportError as e:
    import gurobipy as gp
    HAS_MODELING=False

import numpy as np
from joblib import load
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml2gurobi.extra import morerelu
from ml2gurobi.extra.obbt import obbt
from ml2gurobi.sklearn import (
    Pipe2Gurobi,
    decisiontreeregressor2gurobi,
    gradientboostingregressor2gurobi,
    linearregression2gurobi,
    mlpregressor2gurobi,
    pipe2gurobi,
)


class TestFormulations(unittest.TestCase):
    def fixed_model(self, regressor, translator, X, exampleno, use_gurobi_modeling=False):
        with gp.Model() as m:
            x = m.addMVar(X.shape[1], lb=X[exampleno, :], ub=X[exampleno, :])
            y = m.addMVar(1, lb=-gp.GRB.INFINITY)

            m.Params.OutputFlag = 0
            if use_gurobi_modeling:
                m.add(translator, regressor, x, y)
            else:
                translator(m, regressor, x, y)
            m.optimize()
            m.write('toto.lp')

            self.assertTrue(abs(y.X - regressor.predict(X[exampleno, :].reshape(1, -1))) < 1e-5)

    def test_diabetes(self):
        data = datasets.load_diabetes()

        X = data['data']
        y = data['target']

        to_test = [(LinearRegression(), linearregression2gurobi),
                   (DecisionTreeRegressor(max_leaf_nodes=50), decisiontreeregressor2gurobi),
                   (GradientBoostingRegressor(n_estimators=20), gradientboostingregressor2gurobi),
                   (MLPRegressor([20, 20]), mlpregressor2gurobi)]

        warnings.filterwarnings('ignore')
        for regressor, translator in to_test:
            regressor.fit(X, y)
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, exampleno=exampleno, use_gurobi_modeling=False):
                    self.fixed_model(regressor, translator, X, exampleno)

        if not HAS_MODELING:
            return

        for regressor, translator in to_test:
            regressor.fit(X, y)
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, exampleno=exampleno, use_gurobi_modeling=True):
                    self.fixed_model(regressor, translator, X, exampleno, use_gurobi_modeling=True)


    def test_diabetes_with_pipes(self):
        data = datasets.load_diabetes()

        X = data['data']
        y = data['target']

        to_test = [(LinearRegression(), linearregression2gurobi),
                   (DecisionTreeRegressor(max_leaf_nodes=50), decisiontreeregressor2gurobi),
                   (GradientBoostingRegressor(n_estimators=20), gradientboostingregressor2gurobi),
                   (MLPRegressor([20, 20]), mlpregressor2gurobi)]

        warnings.filterwarnings('ignore')
        for regressor, _ in to_test:
            pipeline = make_pipeline(StandardScaler(), regressor)
            pipeline.fit(X, y)
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, exampleno=exampleno):
                    self.fixed_model(pipeline, pipe2gurobi, X, exampleno)

        if not HAS_MODELING:
            return

        for regressor, _ in to_test:
            pipeline = make_pipeline(StandardScaler(), regressor)
            pipeline.fit(X, y)
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0]-1)
                with self.subTest(regressor=regressor, exampleno=exampleno, use_gurobi_modeling=True):
                    self.fixed_model(pipeline, pipeline, X, exampleno, True)

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

        self.assertEqual(pipe.steps[-1][1].out_activation_, 'identity')
        # Code to add the neural network to the constraints
        pipe2gurobi = Pipe2Gurobi(m, pipe, x, output)
        # For this example we should model softmax in the last layer using identity
        if activation is not None:
            pipe2gurobi.steps[-1].actdict['relu'] = activation
        # pipe2gurobi.steps[-1].actdict['softmax'] = Identity()
        pipe2gurobi.predict()
        return pipe2gurobi

    def test_adversarial_activations(self):
        # Load the trained network and the examples
        dirname = os.path.dirname(__file__)
        pipe = load(os.path.join(dirname, 'networks/MNIST_50_50.joblib'))
        X = load(os.path.join(dirname, 'networks/MNIST_first100.joblib'))

        # Change the out_activation of neural network to identity
        pipe.steps[-1][1].out_activation_ = 'identity'

        # Choose an example
        exampleno = random.randint(0, 99)
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
        dirname = os.path.dirname(__file__)
        pipe = load(os.path.join(dirname, 'networks/MNIST_50_50.joblib'))
        X = load(os.path.join(dirname, 'networks/MNIST_first100.joblib'))

        # Change the out_activation of neural network to identity
        pipe.steps[-1][1].out_activation_ = 'identity'

        # Choose an example
        exampleno = random.randint(0, 99)
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
