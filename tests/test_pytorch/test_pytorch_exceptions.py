import unittest

import gurobipy as gp
import numpy as np
import torch

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import ModelConfigurationError


class TestUnsuportedTorch(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_torch_layers(self):
        """Do a fancy network with lots of layers we don't support"""
        nn_model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=(3, 3), padding="same"),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(10, 20, (3, 3), padding="same"),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.Softmax(1),
        )

        example = np.zeros((1, 28, 28))

        m = gp.Model()

        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")

        with self.assertRaises(ModelConfigurationError):
            add_predictor_constr(m, nn_model, x)
