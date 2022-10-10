import os
import random
import unittest
import warnings

import gurobipy as gp
import numpy as np
from base_cases import DiabetesCases
from gurobipy import GurobiError
from joblib import load
from sklearn import datasets

from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn import PipelineConstr

VERBOSE = False


class TestFixedModel(unittest.TestCase):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def fixed_model(self, predictor, example, nonconvex):
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

            pred_constr = add_predictor_constr(gpm, predictor, x, y)
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
            abserror = np.abs(pred_constr.get_error()).astype(float)
            if abserror > tol:
                gpm.write("Failed.lp")
                print(f"Error: {y.X} != {predictor.predict(example.reshape(1, -1))}")

            self.assertLessEqual(np.abs(pred_constr.get_error()), tol)

    def test_diabetes(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            regressor = onecase["predictor"]
            for _ in range(5):
                exampleno = random.randint(0, X.shape[0] - 1)
                with super().subTest(regressor=regressor, exampleno=exampleno):
                    if VERBOSE:
                        print(f"Doing {regressor} with example {exampleno}")
                    self.fixed_model(regressor, X[exampleno, :].astype(np.float32), onecase["nonconvex"])


class TestReLU(unittest.TestCase):
    """Test that various versions of ReLU work and give the same results."""

    def adversarial_model(self, m, pipe, example, epsilon, activation=None):
        ex_prob = pipe.predict_proba(example)
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

        self.assertEqual(pipe.steps[-1][1].out_activation_, "identity")
        # Code to add the neural network to the constraints
        if activation is not None:
            activation_models = {"relu": activation}
        else:
            activation_models = {}

        pipe2gurobi = PipelineConstr(m, pipe, x, output, activation_models=activation_models)
        # pipe2gurobi.steps[-1].actdict['softmax'] = Identity()
        return pipe2gurobi

    def test_adversarial_activations(self):
        # Load the trained network and the examples
        dirname = os.path.dirname(__file__)
        pipe = load(os.path.join(dirname, "predictors/MNIST_50_50.joblib"))
        X = load(os.path.join(dirname, "predictors/MNIST_first100.joblib"))

        # Change the out_activation of neural network to identity
        pipe.steps[-1][1].out_activation_ = "identity"

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
                    self.adversarial_model(m, pipe, example, epsilon, activation=activation)
                    m.optimize()
                    if value is None:
                        value = m.ObjVal
                    else:
                        self.assertAlmostEqual(value, m.ObjVal, places=5)


if __name__ == "__main__":
    VERBOSE = True
    unittest.main(verbosity=2)
