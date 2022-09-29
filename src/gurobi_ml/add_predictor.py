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
    """Use `predictor` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------
    model: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
            The gurobipy model where the predictor should be inserted.
    predictor:
        The predictor to insert.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    AbstractPredictorConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    The parameters `input_vars` and `output_vars` can be either

     * Gurobipy matrix variables `gp.MVar <https://www.gurobi.com/documentation/9.5/refman/py_mvar.html>`_
     * Lists of variables
     * Dictionaries of variables

    For internal use in the package they are cast into matrix variables and it is
    the prefered and most convenient format to use.

    They should have dimensions that conforms with the input/output of the predictor.
    We denote by `n_features` the dimension of the input of the predictor and `n_output` the dimension of the output.

    If they are lists or a dictionaries `input_vars` should have length `n_features` and `output_vars`
    should have length `n_output`.

    If they are matrix variables, `input_vars` and `output_vars` can be either of shape `(n_features)` and
    `(n_outputs,)` respecitvely or `(k, n_features)` and `(k, n_outputs)` respectively (with `k >= 1`).
    The latter form is espcecially useful if the predictor is used to associate different groups of
    variables (for e.g. a prediction is made for every time period in a planning horizon).
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
