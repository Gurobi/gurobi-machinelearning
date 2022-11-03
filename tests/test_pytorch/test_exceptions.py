import unittest

import gurobipy as gp
import numpy as np
import torch
from tensorflow import keras

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoModel


class TestUnsuportedTorch(unittest.TestCase):
    def setUp(self) -> None:
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
        # convert class vectors to binary classes
        self.y_train = y_train.astype(np.int64)

        # Convert images to float32
        self.x_train = x_train.astype("float32")

    def test_torch_layers(self):
        """Do a fancy network with lots of layers we don't support"""
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.reshape(self.x_train, (-1, 1, 28, 28))
        self.x_train /= 255.0

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

        example = self.x_train[18, :]

        m = gp.Model()

        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, nn_model, x)
