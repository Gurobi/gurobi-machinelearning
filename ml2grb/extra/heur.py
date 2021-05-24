import numpy as np

def prop_layer(layer, input):
    activation_vars = layer.actvar
    layer_coefs = layer.coefs
    layer_intercept = layer.intercept
    activation_function = layer.activation
    assert activation_function in ('relu', 'identity', 'softmax')
    (input_size, _) = layer_coefs.shape
    n = input.shape[0]
    assert input_size == input.shape[1]

    if activation_function == 'relu':
        z = layer.zvar
        assert z is not None
    elif activation_function == 'logit':
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
    n = input.shape[0]
    assert input_size == input.shape[1], f'{input_size} != {input.shape}'

    mixing = input @ layer_coefs + layer_intercept

    # Compute activation
    if activation_function:
        keep_list = activation_function.forward_fixing(layer, mixing, -numfix)
        mixing = activation_function.forward(mixing)
    else:
        keep_list = []
    return (mixing, keep_list)

def most_violated(layer):
    activation_function = layer.activation
    layer_intercept = layer.intercept
    layer_coefs = layer.coefs
    input = layer.invar.X
    activations = layer.actvar.X
    if activation_function != 'relu':
        return None
    input_size = layer_coefs.shape[0]
    n = input.shape[0]
    assert input_size == input.shape[1]

    z = layer.zvar
    assert z is not None
    assert layer.n_partitions == 1
    mixing = input @ layer_coefs + layer_intercept
    relu = np.maximum(mixing, 0.0)
    error = activations - relu
    rval = error.argmax()
    print(f'variable {rval} activation {activations[0, rval]} relu {relu[0,rval]}')
    return (rval, error[0,rval], relu[0, rval])

def relax_act(nn2grb):
    for layer_model in nn2grb._layers:
        layer_model._actvar.LB = layer_model._wmin
        layer_model._actvar.UB = layer_model._wmax
    nn2grb.model.update()

def prop(nn2grb, X, reset=False):
    '''Propagate fixings into the network'''
    # Iterate over the layers
    torelax = list()
    activation = X
    target = 20
    for layer in nn2grb:
        input_vals = activation
        activation, keepopen = prop_activities(layer, input_vals, numfix=min(target, 50))
        torelax += keepopen


    nn2grb.model.optimize()
    nn2grb.canrelax = torelax
    if reset:
        for layer in nn2grb._layers:
            layer.reset_bounds()
            nn2grb.model.update()


def cut_round(nn2grb, model=None):
    nn = nn2grb.regressor

    cuts = list()

    # Iterate over the hidden layers
    for layer in nn2grb:
        if layer.activation is None:
            continue
        cuts += layer_cuts(layer)
    #print(f"Generated {len(cuts)} cuts")
    return cuts
