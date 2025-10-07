import unittest

import numpy as np
import gurobipy as gp
from torch import nn

from gurobi_ml.add_predictor import add_predictor_constr


class TestPytorchCNNAdversarialError(unittest.TestCase):
    def test_get_error_small_time_limit(self):
        """
        Build a tiny ConvNet similar to the adversarial_pytorch_cnn notebook and
        embed it into a Gurobi model. With a very small time limit, solving should
        still return a feasible solution and the predictor embedding should be exact,
        i.e., pred_constr.get_error() is ~0.
        """

        # Tiny, supported CNN: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear
        # Input is NCHW with shape (1, 1, 4, 4)
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2, 3),
        )

        # Create a fixed example input in NHWC used on the MIP side
        # Shape (1, 4, 4, 1)
        rng = np.random.default_rng(0)
        example_nhwc = rng.random((1, 4, 4, 1), dtype=np.float32).astype(float)

        # Set up Gurobi model and variables
        m = gp.Model()
        x = m.addMVar(example_nhwc.shape, lb=0.0, ub=1.0, name="x")
        y = m.addMVar((1, 3), lb=-gp.GRB.INFINITY, name="y")

        # Fix x by setting lower and upper bounds to the same values
        x.lb = example_nhwc
        x.ub = example_nhwc

        # Embed the PyTorch CNN
        pred_constr = add_predictor_constr(m, model, x, y)

        # No real objective needed; keep it simple
        m.setObjective(0.0)

        # Solve with a very small time limit; we only need feasibility
        m.setParam("TimeLimit", 1.0)
        m.optimize()

        self.assertGreater(m.SolCount, 0, "Gurobi did not return a feasible solution.")

        # Check embedding correctness: should be close to exact
        err = pred_constr.get_error()
        self.assertLessEqual(
            float(np.max(err.astype(float))),
            1e-6,
            f"Max error too large: {np.max(err)}",
        )

    def test_flatten_order_matches_pytorch(self):
        """
        Ensure that the flatten ordering in the MIP matches PyTorch's Flatten
        (NCHW -> flatten over C,H,W with W fastest). Use an input where H and W
        are >1 so ordering matters.
        """

        # Build a small CNN that yields a 2x2x1 tensor before Flatten
        # Input: (1, 1, 5, 5) -> Conv(1x1, 1 ch) -> MaxPool(2) -> 2x2x1 -> Flatten(4)
        model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

        rng = np.random.default_rng(1)
        example_nhwc = rng.random((1, 5, 5, 1), dtype=np.float32).astype(float)

        m = gp.Model()
        x = m.addMVar(example_nhwc.shape, lb=0.0, ub=1.0, name="x")
        y = m.addMVar((1, 2), lb=-gp.GRB.INFINITY, name="y")

        # Fix input
        x.lb = example_nhwc
        x.ub = example_nhwc

        pred_constr = add_predictor_constr(m, model, x, y)

        m.setObjective(0.0)
        m.setParam("TimeLimit", 1.0)
        m.optimize()

        self.assertGreater(m.SolCount, 0, "Gurobi did not return a feasible solution.")

        err = pred_constr.get_error()
        self.assertLessEqual(
            float(np.max(err.astype(float))),
            1e-6,
            f"Max error too large: {np.max(err)}",
        )

    def test_flatten_dimensionality_validation_4d_to_2d(self):
        """
        Validate that a 4D NHWC input through Conv2d -> Flatten -> Linear is
        accepted by the validators (accepted_dim handling) and yields near-zero
        get_error when input is fixed via bounds.
        """

        # Model: (1,1,6,6) -> Conv2d(2 filters, k=2) -> (1,2,5,5)
        # -> Flatten (50) -> Linear(50->5)
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(2 * 5 * 5, 5),
        )

        rng = np.random.default_rng(2)
        example_nhwc = rng.random((1, 6, 6, 1), dtype=np.float32).astype(float)

        m = gp.Model()
        x = m.addMVar(example_nhwc.shape, lb=0.0, ub=1.0, name="x")
        y = m.addMVar((1, 5), lb=-gp.GRB.INFINITY, name="y")

        # Fix input via bounds
        x.lb = example_nhwc
        x.ub = example_nhwc

        pred_constr = add_predictor_constr(m, model, x, y)

        m.setObjective(0.0)
        m.setParam("TimeLimit", 1.0)
        m.optimize()

        self.assertGreater(m.SolCount, 0, "Gurobi did not return a feasible solution.")

        err = pred_constr.get_error()
        self.assertLessEqual(
            float(np.max(err.astype(float))),
            1e-6,
            f"Max error too large: {np.max(err)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
