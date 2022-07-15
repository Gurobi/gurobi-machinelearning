from .decisiontrees import (
    DecisionTreeRegressorConstr,
    GradientBoostingRegressorConstr,
    RandomForestRegressorConstr,
)
from .preprocessing import PolynomialFeaturesConstr, StandardScalerConstr
from .regressions import (
    LinearRegressionConstr,
    LogisticRegressionConstr,
    MLPRegressorConstr,
)


def sklearn_transformers():
    return {
        "StandardScaler": StandardScalerConstr,
        "PolynomialFeatures": PolynomialFeaturesConstr,
    }


def sklearn_predictors():
    return {
        "LinearRegression": LinearRegressionConstr,
        "LogisticRegression": LogisticRegressionConstr,
        "DecisionTreeRegressor": DecisionTreeRegressorConstr,
        "GradientBoostingRegressor": GradientBoostingRegressorConstr,
        "RandomForestRegressor": RandomForestRegressorConstr,
        "MLPRegressor": MLPRegressorConstr,
        "MLPClassifier": MLPRegressorConstr,
    }
