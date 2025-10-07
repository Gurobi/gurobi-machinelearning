import unittest

import gurobipy as gp
import numpy as np
import keras

from gurobi_ml.add_predictor import add_predictor_constr
from gurobi_ml.exceptions import ParameterError


class TestKerasCNNFlattenAndInvalid(unittest.TestCase):
    def test_cnn_flatten_4d_to_2d_works(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(6, 6, 1)),
                keras.layers.Conv2D(2, kernel_size=2, strides=1, padding="valid"),
                keras.layers.Flatten(),
                keras.layers.Dense(5),
            ]
        )

        rng = np.random.default_rng(3)
        example_nhwc = rng.random((1, 6, 6, 1), dtype=np.float32).astype(float)

        m = gp.Model()
        x = m.addMVar(example_nhwc.shape, lb=0.0, ub=1.0, name="x")
        y = m.addMVar((1, 5), lb=-gp.GRB.INFINITY, name="y")

        # Fix input
        x.lb = example_nhwc
        x.ub = example_nhwc

        pred = add_predictor_constr(m, model, x, y)

        m.setObjective(0.0)
        m.setParam("TimeLimit", 1.0)
        m.optimize()

        err = pred.get_error()
        self.assertLessEqual(float(np.max(err.astype(float))), 1e-6)

    def test_rejects_5d_input(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(2, 2, 1)),
                keras.layers.Conv2D(1, kernel_size=1),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )

        # 5D shape: (batch=1, H=2, W=2, C=1, D=1)
        bad_shape = (1, 2, 2, 1, 1)
        _ = np.zeros(bad_shape, dtype=float)

        m = gp.Model()
        x = m.addMVar(bad_shape, lb=0.0, ub=1.0, name="x")
        y = m.addMVar((1, 1), lb=-gp.GRB.INFINITY, name="y")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, model, x, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
