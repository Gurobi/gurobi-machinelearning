import os

from joblib import dump, load
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml2gurobi.sklearn import (
    DecisionTreeRegressorConstr,
    GradientBoostingRegressorConstr,
    LinearRegressionConstr,
    MLPRegressorConstr,
)


class DiabetesCases:
    """Base class to have cases for testing regression models on diabetes set"""

    to_test = [
        (LinearRegression(), LinearRegressionConstr),
        (DecisionTreeRegressor(max_leaf_nodes=50), DecisionTreeRegressorConstr),
        (
            GradientBoostingRegressor(n_estimators=20),
            GradientBoostingRegressorConstr,
        ),
        (MLPRegressor([20, 20]), MLPRegressorConstr),
    ]

    def __init__(self):
        self.basedir = os.path.join(os.path.dirname(__file__), "predictors")
        version = None
        with open(os.path.join(self.basedir, "sklearn_version")) as filein:
            version = filein.read().strip()
        if version != sklearn_version:
            print("Scikit learn version changed. Regenerate predictors")
            self.build_predictors()
            with open(os.path.join(self.basedir, "sklearn_version"), "w") as fileout:
                print(sklearn_version, file=fileout)

    def build_predictors(self):
        data = datasets.load_diabetes()

        X = data["data"]
        y = data["target"]

        for predictor, _ in self.to_test:
            predictor.fit(X, y)
            filename = f"diabetes_none_{type(predictor).__name__}.joblib"
            rval = {
                "predictor": predictor,
                "input_shape": X.shape,
                "output_shape": y.shape,
            }

            dump(rval, os.path.join(self.basedir, filename))

        for predictor, _ in self.to_test:
            pipeline = make_pipeline(StandardScaler(), predictor)
            pipeline.fit(X, y)
            filename = f"diabetes_pipe1_{type(predictor).__name__}.joblib"
            rval = {
                "predictor": pipeline,
                "input_shape": X.shape,
                "output_shape": y.shape,
            }
            dump(rval, os.path.join(self.basedir, filename))

        for predictor, _ in self.to_test:
            pipeline = make_pipeline(PolynomialFeatures(), StandardScaler(), predictor)
            pipeline.fit(X, y)
            filename = f"diabetes_pipe2_{type(predictor).__name__}.joblib"
            rval = {
                "predictor": pipeline,
                "input_shape": X.shape,
                "output_shape": y.shape,
            }

            dump(rval, os.path.join(self.basedir, filename))

    def get_case(self, predictor, withpipe):
        if withpipe:
            withpipe = f"pipe{withpipe}"
        else:
            withpipe = "none"
        filename = f"diabetes_{withpipe}_{type(predictor).__name__}.joblib"
        predictor = load(os.path.join(self.basedir, filename))
        return predictor
