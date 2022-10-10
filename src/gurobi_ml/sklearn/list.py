from .decisiontrees import (
    DecisionTreeRegressorConstr,
    GradientBoostingRegressorConstr,
    RandomForestRegressorConstr,
)
from .mlpregressor import MLPRegressorConstr
from .preprocessing import PolynomialFeaturesConstr, StandardScalerConstr
from .regressions import LinearRegressionConstr, LogisticRegressionConstr


def sklearn_transformers():
    return {
        "StandardScaler": StandardScalerConstr,
        "PolynomialFeatures": PolynomialFeaturesConstr,
    }


def sklearn_predictors():
    return {
        "LinearRegression": LinearRegressionConstr,
        "Ridge": LinearRegressionConstr,
        "Lasso": LinearRegressionConstr,
        "LogisticRegression": LogisticRegressionConstr,
        "DecisionTreeRegressor": DecisionTreeRegressorConstr,
        "GradientBoostingRegressor": GradientBoostingRegressorConstr,
        "RandomForestRegressor": RandomForestRegressorConstr,
        "MLPRegressor": MLPRegressorConstr,
        "MLPClassifier": MLPRegressorConstr,
    }
