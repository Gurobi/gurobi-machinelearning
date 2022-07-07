# Copyright © 2022 Gurobi Optimization, LLC
from gurobipy import GRB


def obbt_layer(layer, round_num, stats=None, verbose=True):
    """Perform OBBT on layer"""
    model = layer.getModel()
    obj = model.getObjective()
    objsense = model.ModelSense
    savemethod = model.Params.Method
    input_vars = layer._input
    layer_coefs = layer.coefs
    layer_intercept = layer.intercept

    vtypes = model.getAttr(GRB.Attr.VType, model.getVars())
    model.setAttr(GRB.Attr.VType, model.getVars(), GRB.CONTINUOUS)

    model.optimize()
    assert model.Status == GRB.OPTIMAL
    print(model.NumVars, model.NumConstrs, model.NumGenConstrs)
    if verbose:
        print(f"Round {round_num} objval {model.ObjVal}")
    model.Params.Method = 0

    n = input_vars.shape[0]
    wmin = layer.wmin
    wmax = layer.wmax
    eps = 1e-8
    n_strengthened = 0
    done = 0
    alreadfixed = (wmin > -eps) | (wmax < eps)

    for j in range(layer_coefs.shape[1]):
        w = layer_coefs[:, j]
        w0 = layer_intercept[j]
        for k in range(n):
            if alreadfixed[k, j]:
                continue
            if layer.zvar is not None and (
                layer.zvar[k, j].LB > 0.5 or layer.zvar[k, j].UB < 0.5
            ):
                continue
            done += 1
            model.setObjective(
                w0 + sum(input_vars[k, p] * w[p] for p in range(input_vars.shape[1])),
                GRB.MAXIMIZE,
            )
            model.optimize()
            newbound = model.Objval
            if newbound < wmax[k, j] - 2 * eps:
                n_strengthened += 1
                wmax[k, j] = newbound
            if stats is not None:
                stats["done"] += 1
                stats["iters"] += model.IterCount
    if verbose:
        print(f"OBBT strengthened {n_strengthened} upper bounds on layer (did {done})")

    total_strengthened = n_strengthened
    n_strengthened = 0
    done = 0
    alreadfixed = (wmin > -eps) | (wmax < eps)
    for j in range(layer_coefs.shape[1]):
        w = layer_coefs[:, j]
        w0 = layer_intercept[j]
        for k in range(n):
            if alreadfixed[k, j]:
                continue
            if layer.zvar is not None and (
                layer.zvar[k, j].LB > 0.5 or layer.zvar[k, j].UB < 0.5
            ):
                continue
            done += 1
            model.setObjective(
                w0 + sum(input_vars[k, p] * w[p] for p in range(input_vars.shape[1])),
                GRB.MINIMIZE,
            )
            model.optimize()
            newbound = model.ObjVal
            if newbound > wmin[k, j] + 2 * eps:
                n_strengthened += 1
                wmin[k, j] = newbound
            if stats is not None:
                stats["done"] += 1
                stats["iters"] += model.IterCount
            model.update()

    if verbose:
        print(f"OBBT strengthened {n_strengthened} lower bounds on layer (did {done})")

    # Restore model
    model.Params.Method = savemethod
    model.setAttr(GRB.Attr.VType, model.getVars(), vtypes)
    model.setObjective(obj)
    model.ModelSense = objsense
    model.update()

    return total_strengthened + n_strengthened


def obbt(nn2gurobi, n_rounds=1, activation=None):
    """Perform OBBT on model"""
    stats = {"done": 0, "iters": 0}
    model = nn2gurobi.getModel()
    outputflag = model.Params.OutputFlag
    model.Params.OutputFlag = 0
    nn2gurobi.rebuild_formulation(activation)
    for roundnum in range(n_rounds):
        n_strengthened = 0
        for layer in nn2gurobi:
            if layer.activation in (None, "identity"):
                continue
            n_strengthened += obbt_layer(
                layer, roundnum, stats=stats, verbose=outputflag
            )
            nn2gurobi.rebuild_formulation(activation)
        if n_strengthened == 0:
            break
    nn2gurobi.rebuild_formulation()
    model.Params.OutputFlag = outputflag
    return stats
