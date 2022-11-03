import os

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


class Cases:
    """Base class to have cases for testing"""

    def __init__(self, dataset, excluded=None, regressors=None, transformers=None):
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")
        self.dataset = dataset

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
            self.build_predictors()
            with open(os.path.join(self.basedir, sklearn_version_file), "w") as file_out:
                print(sklearn_version, file=file_out)

    def __iter__(self):
        return self.all_test.__iter__()

    def get_case(self, predictor):
        filename = f"{self.dataset}_{predictor_as_string(predictor)}.joblib"
        predictor = load(os.path.join(self.basedir, filename))
        return predictor


class DiabetesCases(Cases):
    """Base class to have cases for testing regression models on diabetes set"""

    def __init__(self):
        excluded = ["LogisticRegression"]
        super().__init__("diabetes", excluded=excluded)
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")

    def build_predictors(self):
        data = datasets.load_diabetes()

        X = data["data"]
        y = data["target"]
        for predictor in self:
            predictor.fit(X, y)
            filename = f"{self.dataset}_{predictor_as_string(predictor)}.joblib"
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

            dump(rval, os.path.join(self.basedir, filename))
