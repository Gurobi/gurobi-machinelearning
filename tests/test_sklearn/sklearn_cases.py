import os
from abc import ABC, abstractmethod

import numpy as np
from joblib import dump, load
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor  # noqa
from sklearn.ensemble import RandomForestRegressor  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa
from sklearn.pipeline import Pipeline  # noqa
from sklearn.pipeline import make_pipeline  # noqa
from sklearn.preprocessing import PolynomialFeatures  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
from sklearn.tree import DecisionTreeRegressor  # noqa

from gurobi_ml.sklearn import sklearn_predictors, sklearn_transformers


def predictor_params(name):
    if name == "MLPRegressor":
        return "[20, 20]"
    if name == "MLPClassifier":
        return "[50, 50]"
    if name == "GradientBoostingRegressor":
        return "n_estimators=10, max_depth=4, max_leaf_nodes=10"
    if name == "RandomForestRegressor":
        return "n_estimators=10, max_depth=4, max_leaf_nodes=10"
    if name == "DecisionTreeRegressor":
        return "max_leaf_nodes=50"
    return ""


def init_predictor(name):
    params = predictor_params(name)
    return eval(f"{name}({params})")


def predictor_as_string(predictor):
    rval = ""
    if isinstance(predictor, Pipeline):
        rval += "_pipeline"
        for predictor in predictor:
            rval += predictor_as_string(predictor)
        return rval
    if isinstance(predictor, MLPRegressor):
        size = ""
        nn = predictor
        for s in nn.hidden_layer_sizes[:-1]:
            size += f"{s}x"
        size += f"{nn.hidden_layer_sizes[-1]}"
        rval += f"_mplregressor_{size}"
        return rval
    return "_" + type(predictor).__name__.lower()


class Cases(ABC):
    """Base class to have cases for testing

    This class is used to construct and save the predictors we want to test.

    It is generic in that it doesn't specify the data used.

    Attributes
    ----------
    excluded: list
    List of sklearn predictors that we don't want to test in a specific case

    regressors: list, optional
    List of sklearn regression models that we want to test for a specific a case.
    With default value None, test all predictors we know about.

    transformers: list, optional
    List of sklearn preprocessing objects that we want to test for a specific case.
    With default value None, test all preprocessing we know about.

    saved_training: int, optional
    Number of training example we want to save with the predictor.
    With default value of 0, don't save anything.
    """

    def __init__(
        self, dataset, excluded=None, regressors=None, transformers=None, saved_training=0
    ):
        self.basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")
        self.dataset = dataset
        self.saved_training = saved_training

        # Filled with get data if needed
        self._data = None

        if regressors is None:
            regressors = [r for r in sklearn_predictors().keys() if r not in excluded]
        if transformers is None:
            transformers = list(sklearn_transformers().keys())

        self.all_test = [init_predictor(reg) for reg in regressors]

        if len(transformers):
            self.all_test += [
                make_pipeline(init_predictor(trans), init_predictor(reg))
                for trans in transformers
                for reg in regressors
            ]
        else:
            self.all_test += [make_pipeline(init_predictor(reg)) for reg in regressors]
        sklearn_version_file = f"{dataset}_sklearn_version"

        try:
            with open(os.path.join(self.basedir, sklearn_version_file)) as file_in:
                version = file_in.read().strip()
        except FileNotFoundError:
            version = None
        if version != sklearn_version:
            print(f"Scikit learn version changed. Regenerate predictors for {dataset}")
            self.build_all_predictors()
            with open(os.path.join(self.basedir, sklearn_version_file), "w") as file_out:
                print(sklearn_version, file=file_out)

    def __iter__(self):
        return self.all_test.__iter__()

    @abstractmethod
    def load_data(self):
        """Define this to load data for predictors"""
        ...

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    def predictor_file(self, predictor):
        return f"{self.dataset}_{predictor_as_string(predictor)}.joblib"

    def build_predictor(self, predictor):
        """Build model for one predictor"""
        X, y = self.data
        predictor.fit(X, y)
        non_convex = False
        if isinstance(predictor, Pipeline):
            for element in predictor:
                if isinstance(element, PolynomialFeatures):
                    non_convex = True
                    break

        rval = {
            "predictor": predictor,
            "input_shape": X.shape,
            "output_shape": y.shape,
            "nonconvex": non_convex,
        }
        if self.saved_training:
            rval["data"] = X[: self.saved_training]
            rval["target"] = X[: self.saved_training]
        dump(rval, os.path.join(self.basedir, self.predictor_file(predictor)))
        return rval

    def build_all_predictors(self):
        """Build all the predictor for this case.
        (Done when we have a new sklearn version)"""
        for predictor in self:
            rval = self.build_predictor(predictor)

            dump(rval, os.path.join(self.basedir, self.predictor_file(predictor)))

    def get_case(self, predictor):
        filename = self.predictor_file(predictor)
        try:
            return load(os.path.join(self.basedir, filename))
        except FileNotFoundError:
            return self.build_predictor(predictor)


class DiabetesCases(Cases):
    """Base class to have cases for testing regression models on diabetes set

    This is appropriate for testing a regression with a single output."""

    def __init__(self):
        excluded = ["LogisticRegression"]
        super().__init__("diabetes", excluded=excluded)

    def load_data(self):
        data = datasets.load_diabetes()

        X = data["data"]
        y = data["target"]
        self._data = (X, y)


class IrisCases(Cases):
    """Base class to have cases for testing regression models on iris set

    Transform the iris test set to binary classification.
    This is appropriate for testing binary classification models."""

    def __init__(self):
        super().__init__("iris", regressors=["LogisticRegression"])

    def load_data(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target
        # Make it a binary classification
        X = X[y != 2]
        y = y[y != 2]
        self._data = (X, y)


class CircleCase(Cases):
    """Artificial test case to test multi-output regression models

    Currently we use it for decision trees and random forests."""

    def __init__(self):
        super().__init__(
            "circle",
            regressors=["DecisionTreeRegressor", "RandomForestRegressor"],
            saved_training=-1,
        )

    def load_data(self):
        # Inspired bu Scikit-learn example
        # Create a dataset drawing a circle (don't put noise at it's not
        # really relevant here)
        rng = np.random.RandomState(1)
        X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
        y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
        self._data = (X, y)


class MNISTCase(Cases):
    """MNIST test case

    We use it for multi output neural networks regressions."""

    def __init__(self):
        super().__init__(
            "mnist",
            regressors=[
                "MLPClassifier",
            ],
            transformers=[],
            saved_training=100,
        )

    def load_data(self):
        mnist = datasets.fetch_openml("mnist_784")
        X, y = mnist.data, mnist.target

        X = X.to_numpy()
        y = y.to_numpy()
        X /= 255.0  # scaling
        self._data = (X, y)
