import os

from joblib import dump, load
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor  # noqa
from sklearn.ensemble import RandomForestRegressor  # noqa
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


class DiabetesCases:
    """Base class to have cases for testing regression models on diabetes set"""

    def __init__(self):
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")
        version = None

        excluded = ["LogisticRegression", "MLPClassifier"]

        regressors = [r for r in sklearn_predictors().keys() if r not in excluded]
        transformers = list(sklearn_transformers().keys())

        self.all_test = [init_predictor(reg) for reg in regressors]

        self.all_test += [
            make_pipeline(init_predictor(trans), init_predictor(reg)) for trans in transformers for reg in regressors
        ]
        print(self.all_test)
        with open(os.path.join(self.basedir, "sklearn_version")) as filein:
            version = filein.read().strip()
        if version != sklearn_version:
            print("Scikit learn version changed. Regenerate predictors")
            self.build_predictors()
            with open(os.path.join(self.basedir, "sklearn_version"), "w") as fileout:
                print(sklearn_version, file=fileout)

    def __iter__(self):
        return self.all_test.__iter__()

    def build_predictors(self):
        data = datasets.load_diabetes()

        X = data["data"]
        y = data["target"]
        for predictor in self:
            predictor.fit(X, y)
            filename = f"diabetes_{predictor_as_string(predictor)}.joblib"
            nonconvex = False
            if isinstance(predictor, Pipeline):
                for element in predictor:
                    if isinstance(element, PolynomialFeatures):
                        nonconvex = True
                        break

            rval = {"predictor": predictor, "input_shape": X.shape, "output_shape": y.shape, "nonconvex": nonconvex}

            dump(rval, os.path.join(self.basedir, filename))

    def get_case(self, predictor):
        filename = f"diabetes_{predictor_as_string(predictor)}.joblib"
        predictor = load(os.path.join(self.basedir, filename))
        return predictor
