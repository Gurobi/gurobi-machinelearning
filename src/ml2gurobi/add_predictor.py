""" Define generic function that can add any known trained predictor
"""
HASSKLEARN = False
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    HASSKLEARN = True
except ImportError:
    pass

if HASSKLEARN:
    from .sklearn import (
        DecisionTreeRegressorConstr,
        GradientBoostingRegressorConstr,
        LinearRegressionConstr,
        LogisticRegressionConstr,
        MLPRegressorConstr,
        PipelineConstr,
        PolynomialFeaturesConstr,
        RandomForestRegressorConstr,
        StandardScalerConstr,
    )

HASPYTORCH = False
try:
    from pytorch import nn as pytorchnn

    HASPYTORCH = True
except ImportError:
    pass

if HASPYTORCH:
    from .pytorch import SequentialConstr


def sklearn_convertors():
    """Collect known convertors for scikit learn objects"""
    if HASSKLEARN:
        return {
            StandardScaler: StandardScalerConstr,
            LinearRegression: LinearRegressionConstr,
            LogisticRegression: LogisticRegressionConstr,
            PolynomialFeatures: PolynomialFeaturesConstr,
            DecisionTreeRegressor: DecisionTreeRegressorConstr,
            GradientBoostingRegressor: GradientBoostingRegressorConstr,
            RandomForestRegressor: RandomForestRegressorConstr,
            MLPRegressor: MLPRegressorConstr,
            Pipeline: PipelineConstr,
        }
    return {}


def pytorch_convertors():
    """Collect known convertors for pytorch objects"""
    if HASPYTORCH:
        return {pytorchnn.Sequential: SequentialConstr}
    return {}


def add_predictor_constr(model, predictor, input_vars, output_vars=None):
    """Use predictor to make the value of y predicted from the values of x
    in the model.
    """
    convertors = {}
    convertors |= sklearn_convertors()
    convertors |= pytorch_convertors()
    for predictor_type, convertor in convertors.items():
        if isinstance(predictor, predictor_type):
            return convertor(model, predictor, input_vars, output_vars)
    raise BaseException(f"No converter for predictor {predictor}")
