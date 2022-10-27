import os

import numpy as np
from joblib import dump, load
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor  # noqa
from sklearn.ensemble import RandomForestRegressor  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa
from sklearn.neural_network import MLPRegressor  # noqa
from sklearn.pipeline import Pipeline  # noqa
from sklearn.pipeline import make_pipeline  # noqa
from sklearn.preprocessing import PolynomialFeatures  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
from sklearn.tree import DecisionTreeRegressor  # noqa

from gurobi_ml.sklearn import sklearn_predictors, sklearn_transformers


def predictor_params(name):
    if name == "MLPRegressor":
        return "[20, 20]"
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


class Cases:
    """Base class to have cases for testing"""

    def __init__(self, excluded):
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")
        version = None

        regressors = [r for r in sklearn_predictors().keys() if r not in excluded]
        transformers = list(sklearn_transformers().keys())

        self.all_test = [init_predictor(reg) for reg in regressors]

        self.all_test += [
            make_pipeline(init_predictor(trans), init_predictor(reg)) for trans in transformers for reg in regressors
        ]
        with open(os.path.join(self.basedir, "sklearn_version")) as filein:
            version = filein.read().strip()
        if version != sklearn_version:
            print("Scikit learn version changed. Regenerate predictors")
            self.build_predictors()
            with open(os.path.join(self.basedir, "sklearn_version"), "w") as fileout:
                print(sklearn_version, file=fileout)

    def __iter__(self):
        return self.all_test.__iter__()

    def get_case(self, predictor):
        filename = f"{self.dataset}_{predictor_as_string(predictor)}.joblib"
        predictor = load(os.path.join(self.basedir, filename))
        return predictor


class DiabetesCases(Cases):
    """Base class to have cases for testing regression models on diabetes set"""

    def __init__(self):
        excluded = ["LogisticRegression", "MLPClassifier"]
        self.dataset = "diabetes"
        super().__init__(excluded=excluded)
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")

    def build_predictors(self):
        data = datasets.load_diabetes()

        X = data["data"]
        y = data["target"]
        for predictor in self:
            predictor.fit(X, y)
            filename = f"{self.dataset}_{predictor_as_string(predictor)}.joblib"
            nonconvex = False
            if isinstance(predictor, Pipeline):
                for element in predictor:
                    if isinstance(element, PolynomialFeatures):
                        nonconvex = True
                        break

            rval = {"predictor": predictor, "input_shape": X.shape, "output_shape": y.shape, "nonconvex": nonconvex}

            dump(rval, os.path.join(self.basedir, filename))


class IrisCases(Cases):
    """Base class to have cases for testing regression models on diabetes set"""

    def __init__(self):
        excluded = [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "DecisionTreeRegressor",
            "GradientBoostingRegressor",
            "RandomForestRegressor",
            "MLPRegressor",
            "MLPClassifier",
        ]
        self.dataset = "iris"
        super().__init__(excluded=excluded)
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")

    def build_predictors(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        # Make it a simple classification
        X = X[y != 2]
        y = y[y != 2]

        for predictor in self:
            predictor.fit(X, y)
            filename = f"{self.dataset}_{predictor_as_string(predictor)}.joblib"
            nonconvex = False
            if isinstance(predictor, Pipeline):
                for element in predictor:
                    if isinstance(element, PolynomialFeatures):
                        nonconvex = True
                        break

            rval = {"predictor": predictor, "input_shape": X.shape, "output_shape": y.shape, "nonconvex": nonconvex}

            dump(rval, os.path.join(self.basedir, filename))


class CircleCase(Cases):
    def __init__(self):
        excluded = [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "LogisticRegression",
            "GradientBoostingRegressor",
            "RandomForestRegressor",
            "MLPRegressor",
        ]
        self.dataset = "circle"
        super().__init__(excluded=excluded)
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")

    def build_predictors(self):
        # Inspired bu Scikit-learn example
        # Create a dataset drawing a circle (don't put noise at it's not
        # really relevant here)
        rng = np.random.RandomState(1)
        X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
        y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T

        for predictor in self:
            predictor.fit(X, y)
            filename = f"{self.dataset}_{predictor_as_string(predictor)}.joblib"
            nonconvex = False
            if isinstance(predictor, Pipeline):
                for element in predictor:
                    if isinstance(element, PolynomialFeatures):
                        nonconvex = True
                        break

            rval = {
                "predictor": predictor,
                "input_shape": X.shape,
                "output_shape": y.shape,
                "nonconvex": nonconvex,
                "data": X,
                "target": y,
            }

            dump(rval, os.path.join(self.basedir, filename))
