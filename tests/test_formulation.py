import os
import random
import unittest
import warnings

import gurobipy as gp
import numpy as np
import tensorflow as tf
import torch
from base_cases import DiabetesCases, IrisCases
from gurobipy import GurobiError
from joblib import load
from sklearn import datasets

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoSolution
from gurobi_ml.sklearn import add_mlp_regressor_constr

VERBOSE = False


class TestFixedModel(unittest.TestCase):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def fixed_model(self, predictor, example, nonconvex, isproba=False):
        params = {
            "OutputFlag": 0,
            "NonConvex": 2,
        }
        for param in params:
            try:
                params[param] = int(params[param])
            except ValueError:
                pass

        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(example.shape, lb=example - 1e-4, ub=example + 1e-4)
            y = gpm.addMVar(1, lb=-gp.GRB.INFINITY)

            pred_constr = add_predictor_constr(gpm, predictor, x, y, epsilon=1e-5, float_type=np.float32)
            with self.assertRaises(NoSolution):
                pred_constr.get_error()
            with open(os.devnull, "w") as outnull:
                pred_constr.print_stats(file=outnull)
            try:
                gpm.optimize()
            except GurobiError as E:
                if E.errno == 10010:
                    warnings.warn(UserWarning("Limited license"))
                    self.skipTest("Model too large for limited license")
                else:
                    raise

            if nonconvex:
                tol = 5e-3
            else:
                tol = 1e-5
            vio = gpm.MaxVio
            if vio > 1e-5:
                warnings.warn(UserWarning(f"Big solution violation {vio}"))
                warnings.warn(UserWarning(f"predictor {predictor}"))
            tol = max(tol, vio)
            tol *= np.abs(pred_constr.output.X)
            if isproba:
                abserror = np.abs(pred_constr.get_error_proba()).astype(float)
            else:
                abserror = np.abs(pred_constr.get_error()).astype(float)
            if (abserror > tol).any():
                if isproba:
                    print(f"Error: {y.X} != {predictor.predict_proba(example.reshape(1, -1))}")
                else:
                    print(f"Error: {y.X} != {predictor.predict(example.reshape(1, -1))}")

            self.assertLessEqual(np.max(abserror), tol)

    def all_of_them(self):  # pragma: no cover
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        onecase = cases.get_case(cases.all_test[17])
        regressor = onecase["predictor"]
        for exampleno in range(X.shape[0]):
            with super().subTest(regressor=regressor, exampleno=exampleno):
                print("Test {}".format(exampleno))
                self.fixed_model(regressor, X[exampleno, :].astype(np.float32), onecase["nonconvex"])

    def test_diabetes(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            regressor = onecase["predictor"]
            for _ in range(1):
                exampleno = random.randint(0, X.shape[0] - 1)
                with super().subTest(regressor=regressor, exampleno=exampleno):
                    if VERBOSE:
                        print(f"Doing {regressor} with example {exampleno}")
                    self.fixed_model(regressor, X[exampleno, :].astype(np.float32), onecase["nonconvex"])

    def test_diabetes_pytorch(self):
        data = datasets.load_diabetes()

        X = data["data"]

        filename = os.path.join(os.path.dirname(__file__), "predictors", "diabetes__pytorch.pt")
        regressor = torch.load(filename)
        for _ in range(1):
            exampleno = random.randint(0, X.shape[0] - 1)
            with super().subTest(regressor=regressor, exampleno=exampleno):
                if VERBOSE:
                    print(f"Doing {regressor} with example {exampleno}")
                self.fixed_model(regressor, X[exampleno, :].astype(np.float32), 0)

    def test_diabetes_keras(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        filename = os.path.join(os.path.dirname(__file__), "predictors", "diabetes_keras")
        regressor = tf.keras.models.load_model(filename)
        for _ in range(1):
            exampleno = random.randint(0, X.shape[0] - 1)
            with super().subTest(regressor=regressor, exampleno=exampleno):
                if VERBOSE:
                    print(f"Doing {regressor} with example {exampleno}")
                self.fixed_model(regressor, X[exampleno, :].astype(np.float32), 0)

    def test_diabetes_keras_alt(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        filename = os.path.join(os.path.dirname(__file__), "predictors", "diabetes_keras_v2")
        regressor = tf.keras.models.load_model(filename)
        for _ in range(1):
            exampleno = random.randint(0, X.shape[0] - 1)
            with super().subTest(regressor=regressor, exampleno=exampleno):
                if VERBOSE:
                    print(f"Doing {regressor} with example {exampleno}")
                self.fixed_model(regressor, X[exampleno, :].astype(np.float32), 0)

    def test_iris(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        # Make it a simple classification
        X = X[y != 2]
        y = y[y != 2]
        cases = IrisCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            regressor = onecase["predictor"]
            for _ in range(1):
                exampleno = random.randint(0, X.shape[0] - 1)
                with super().subTest(regressor=regressor, exampleno=exampleno):
                    if VERBOSE:
                        print(f"Doing {regressor} with example {exampleno}")
                    self.fixed_model(regressor, X[exampleno, :].astype(np.float32), onecase["nonconvex"], 1)


class TestReLU(unittest.TestCase):
    """Test that various versions of ReLU work and give the same results."""

    def adversarial_model(self, m, nn, example, epsilon, activation=None):
        ex_prob = nn.predict_proba(example)
        output_shape = ex_prob.shape
        sortedidx = np.argsort(ex_prob)[0]

        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="X")
        absdiff = m.addMVar(example.shape, lb=0, ub=1, name="dplus")
        output = m.addMVar(output_shape, lb=-gp.GRB.INFINITY, name="y")

        m.setObjective(output[0, sortedidx[-2]] - output[0, sortedidx[-1]], gp.GRB.MAXIMIZE)

        # Bound on the distance to example in norm-2
        m.addConstr(absdiff[0, :] >= x[0, :] - example.to_numpy()[0, :])
        m.addConstr(absdiff[0, :] >= -x[0, :] + example.to_numpy()[0, :])
        m.addConstr(absdiff[0, :].sum() <= epsilon)

        self.assertEqual(nn.out_activation_, "identity")
        # Code to add the neural network to the constraints
        if activation is not None:
            activation_models = {"relu": activation}
        else:
            activation_models = {}

        nn_constr = add_mlp_regressor_constr(m, nn, x, output, activation_models=activation_models)
        # pipe2gurobi.steps[-1].actdict['softmax'] = Identity()
        return nn_constr

    def test_adversarial_sklearn(self):
        # Load the trained network and the examples
        dirname = os.path.dirname(__file__)
        nn = load(os.path.join(dirname, "predictors/MNIST_50_50.joblib"))
        X = load(os.path.join(dirname, "predictors/MNIST_first100.joblib"))

        # Change the out_activation of neural network to identity
        nn.out_activation_ = "identity"

        # Choose an example
        exampleno = random.randint(0, 99)
        example = X.iloc[exampleno : exampleno + 1, :]
        epsilon = 0.01
        activations = (None,)
        value = None
        for activation in activations:
            with gp.Model() as m:
                m.Params.OutputFlag = 0
                with self.subTest(
                    example=exampleno,
                    epsilon=epsilon,
                    activation=activation,
                    obbt=False,
                ):
                    self.adversarial_model(m, nn, example, epsilon, activation=activation)
                    m.optimize()
                    if value is None:
                        value = m.ObjVal
                    else:
                        self.assertAlmostEqual(value, m.ObjVal, places=5)


if __name__ == "__main__":
    VERBOSE = True
    unittest.main(verbosity=2)
