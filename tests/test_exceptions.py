import unittest

import gurobipy as gp
import numpy as np
import tensorflow as tf
import torch
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVR
from tensorflow import keras

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoModel, ParameterError


class TestUnsuportedKeras(unittest.TestCase):
    def setUp(self) -> None:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # convert class vectors to binary classes
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)

        # Convert images to float32
        self.x_train = x_train.astype("float32")
        self.x_test = x_test.astype("float32")

    def do_test(self, nn):
        nn.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        nn.fit(
            self.x_train,
            self.y_train,
            epochs=0,
            validation_data=(self.x_test, self.y_test),
        )

        example = self.x_train[18, :]

        m = gp.Model()

        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, nn, x)

    def test_keras_bad_activation(self):
        """Do a dense network with a bad activation"""
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.reshape(self.x_train, (-1, 28 * 28))
        self.x_test = np.reshape(self.x_test, (-1, 28 * 28))

        nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(28 * 28),
                tf.keras.layers.Dense(50, activation="sigmoid"),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )
        self.do_test(nn)

    def test_keras_layers(self):
        """Do a fancy network with lots of layers we don't support"""
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.reshape(self.x_train, (-1, 28, 28, 1))
        self.x_test = np.reshape(self.x_test, (-1, 28, 28, 1))

        nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer((28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        self.do_test(nn)

    def do_relu_tests(self, **kwargs):
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.reshape(self.x_train, (-1, 28 * 28))
        self.x_test = np.reshape(self.x_test, (-1, 28 * 28))

        nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(28 * 28),
                tf.keras.layers.Dense(50),
                tf.keras.layers.ReLU(**kwargs),
                tf.keras.layers.Dense(10),
            ]
        )
        self.do_test(nn)

    def test_negative_slope(self):
        self.do_relu_tests(negative_slope=1.0)

    def test_threshold(self):
        self.do_relu_tests(threshold=1.0)

    def test_max_value(self):
        self.do_relu_tests(max_value=10.0)


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


class TestUnsuportedSklearn(unittest.TestCase):
    def test_logistic_multiclass(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, logreg, x)

    def test_logistic_wrongarg(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, x, output_type="proba")

    def test_mlpregressor_wrong_act(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = MLPRegressor([10] * 2, activation="logistic")
        mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_pipeline_fail_transformer(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = make_pipeline(QuantileTransformer(), MLPRegressor([10] * 2))
        mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_pipeline_fail_regression(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = make_pipeline(LinearSVR())
        mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)
