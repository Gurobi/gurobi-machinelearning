import unittest

import gurobipy as gp
import numpy as np
from torch import nn

from gurobi_ml.add_predictor import add_predictor_constr
from gurobi_ml.exceptions import ParameterError


class TestPytorchInvalidDims(unittest.TestCase):
    def test_rejects_5d_input(self):
        """Passing a 5D input tensor to a CNN Sequential should raise ParameterError."""
        model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(1 * 1 * 1, 1),
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
