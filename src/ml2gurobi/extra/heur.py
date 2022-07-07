# Copyright © 2022 Gurobi Optimization, LLC
import numpy as np


def prop_layer(layer, input):
    activation_vars = layer.actvar
    layer_coefs = layer.coefs
    layer_intercept = layer.intercept
    activation_function = layer.activation
    assert activation_function in ("relu", "identity", "softmax")
    (input_size, _) = layer_coefs.shape
    assert input_size == input.shape[1]

    if activation_function == "relu":
        z = layer.zvar
        assert z is not None
    elif activation_function == "logit":
        z = layer.zvar
        assert z is not None
    else:
        pass

    mixing = input @ layer_coefs + layer_intercept

    # Compute activation
    if activation_function:
        activation = activation_function.forward(mixing)

        activation_vars.LB = activation
        activation_vars.UB = activation
    return activation


def prop_activities(layer, input, numfix=20):
    layer_coefs = layer.coefs
    layer_intercept = layer.intercept
    activation_function = layer.activation
    input_size = layer_coefs.shape[0]
    assert input_size == input.shape[1], f"{input_size} != {input.shape}"

    mixing = input @ layer_coefs + layer_intercept

    # Compute activation
    if activation_function:
        activation_function.forward_fixing(layer, mixing, -numfix)
        mixing = activation_function.forward(mixing)
    return mixing


def most_violated(layer):
    activation_function = layer.activation
    layer_intercept = layer.intercept
    layer_coefs = layer.coefs
    input = layer.invar.X
    activations = layer.actvar.X
    if activation_function != "relu":
        return None
    input_size = layer_coefs.shape[0]
    assert input_size == input.shape[1]

    z = layer.zvar
    assert z is not None
    assert layer.n_partitions == 1
    mixing = input @ layer_coefs + layer_intercept
    relu = np.maximum(mixing, 0.0)
    error = activations - relu
    rval = error.argmax()
    print(f"variable {rval} activation {activations[0, rval]} relu {relu[0,rval]}")
    return (rval, error[0, rval], relu[0, rval])


def relax_act(nn2gurobi):
    for layer_model in nn2gurobi._layers:
        layer_model._actvar.LB = layer_model._wmin
        layer_model._actvar.UB = layer_model._wmax
    nn2gurobi.model.update()


def prop(nn2gurobi, X, reset=True):
    """Propagate fixings into the network"""
    # Iterate over the layers
    activation = X
    for layer in nn2gurobi:
        input_vals = activation
        activation = prop_activities(layer, input_vals)

    nn2gurobi.model.optimize()
    if reset:
        nn2gurobi.reset_bounds()
