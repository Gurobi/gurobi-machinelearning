""" Define generic function that can add any known trained predictor
"""
from .sklearn import PipelineConstr, sklearn_predictors, sklearn_transformers

HASPYTORCH = False
try:
    from pytorch import nn as pytorchnn

    HASPYTORCH = True
except ImportError:
    pass

if HASPYTORCH:
    from .pytorch import Sequential as TorchSequential

HASTFKERAS = False
try:
    from keras.engine.functional import Functional
    from keras.engine.training import Model
    from tensorflow import keras

    HASTFKERAS = True
except ImportError:
    pass

if HASTFKERAS:
    from .keras import Predictor as KerasPredictor

USER_PREDICTORS = {}


def sklearn_convertors():
    """Collect known convertors for scikit learn objects"""
    return (
        sklearn_transformers()
        | sklearn_predictors()
        | {
            "Pipeline": PipelineConstr,
        }
    )


def pytorch_convertors():
    """Collect known convertors for pytorch objects"""
    if HASPYTORCH:
        return {pytorchnn.Sequential: TorchSequential}
    return {}


def keras_convertors():
    if HASTFKERAS:
        return {keras.Sequential: KerasPredictor, Functional: KerasPredictor, Model: KerasPredictor}
    return {}


def register_predictor_constr(predictor, predictor_constr):
    USER_PREDICTORS[predictor] = predictor_constr


def add_predictor_constr(model, predictor, input_vars, output_vars=None, **kwargs):
    """Use predictor to make the value of y predicted from the values of x
    in the model.
    """
    convertors = {}
    convertors |= sklearn_convertors()
    convertors |= pytorch_convertors()
    convertors |= keras_convertors()
    convertors |= USER_PREDICTORS
    try:
        convertor = convertors[type(predictor)]
    except KeyError:
        convertor = None
    for parent in type(predictor).mro():
        try:
            convertor = convertors[parent]
            break
        except KeyError:
            pass
    if convertor is None:
        try:
            convertor = convertors[type(predictor).__name__]
        except:
            raise BaseException(f"No converter for predictor {predictor}")
    return convertor(model, predictor, input_vars, output_vars, **kwargs)
