""" Define generic function that can add any known trained predictor
"""
import sys

from .sklearn import PipelineConstr, sklearn_predictors, sklearn_transformers

USER_PREDICTORS = {}


def sklearn_convertors():
    """Collect known scikit-learn objects that can be embeded and the conversion class"""
    return (
        sklearn_transformers()
        | sklearn_predictors()
        | {
            "Pipeline": PipelineConstr,
        }
    )


def pytorch_convertors():
    """Collect known PyTorch objects that can be embeded and the conversion class"""
    if "pytorch" in sys.modules:
        from pytorch import nn as pytorchnn

        from .pytorch import Sequential as TorchSequential

        return {pytorchnn.Sequential: TorchSequential}
    return {}


def keras_convertors():
    """Collect known Keras objects that can be embeded and the conversion class"""
    if "tensorflow" in sys.modules:
        from keras.engine.functional import Functional
        from keras.engine.training import Model
        from tensorflow import keras

        from .keras import Predictor as KerasPredictor

        return {keras.Sequential: KerasPredictor, Functional: KerasPredictor, Model: KerasPredictor}
    return {}


def register_predictor_constr(predictor, predictor_constr):
    """Register a new perdictor that can be added using use_predictor_constr

    Parameters
    ----------
    predictor:
        Class of the predictor
    predictor_constr:
        Class implementing the MIP model that embeds a trained object of
        class predictor in a gurobi Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>
    """
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
