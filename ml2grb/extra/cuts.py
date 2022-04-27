import numpy as np
import gurobipy as gp
from gurobipy import GRB


def layer_cuts(layer, model=None):
    layer_intercept = layer.intercept
    layer_coefs = layer.coefs
    input_vars = layer.invar
    activation_vars = layer.actvar
    cuts = list()
    if model:
        X = model.cbGetNodeRel(input_vars)
    else:
        X = input_vars.X

    assert layer.zvar is not None
    LB = input_vars.LB
    UB = input_vars.UB

    for j in range(activation_vars.shape[1]):
        # Go neuron by neuron
        w = layer_coefs[:, j]
        w0 = layer_intercept[j]
        for k in range(1):
            z = layer.zvar[k, j]
            out = activation_vars[k, j]

            if model:
                outX = model.cbGetNodeRel(out)
                zX = model.cbGetNodeRel(z)
            else:
                outX = out.X
                zX = z.X

            if (w == 0).all():
                continue
            Lstar = LB[k, :].copy()
            Ustar = UB[k, :].copy()
            negw = w < 0.0
            Lstar[negw] = UB[k, :][negw]
            Ustar[negw] = LB[k, :][negw]
            Istar = w*X[k, :] < w*(Lstar + zX*(Ustar - Lstar))

            viol = (outX -
                    (zX*(w0 + w[~Istar]@Ustar[~Istar] + w[Istar]@Lstar[Istar]) +
                     w[Istar]@X[k, Istar] - w[Istar]@Lstar[Istar]))

            if viol > 1e-5:
                cuts.append(out <= z*(w0 + w[~Istar]@Ustar[~Istar] + w[Istar]@Lstar[Istar]) +
                            gp.LinExpr(w[Istar], input_vars[k, Istar].tolist()) - w[Istar] @ Lstar[Istar])
    return cuts


def cut_round(nn2grb, model=None):
    cuts = list()

    # Iterate over the hidden layers
    for layer in nn2grb:
        if layer.activation is None:
            continue
        cuts += layer_cuts(layer)
    return cuts


def ReLUcb(model, where):
    '''Generate cuts for the ReLU activation in the network
    Also try to compute a feasible solution by forward propagation'''
    if where != GRB.Callback.MIPNODE:
        return
    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return
    print(f"CB nodes {model.cbGet(GRB.Callback.MIPNODE_NODCNT)}")

    nn2grb = model._nn2grb
    cuts = nn2grb.cut_round(model)
    if cuts is None:
        return
    for c in cuts:
        model.cbCut(c)
        model._mycuts.append(c)


def make_lastrows_cuts(model, startcuts):
    Lazy = np.array(model.getAttr(GRB.Attr.Lazy, model.getConstrs()))
    Lazy[startcuts:] = -1
    model.setAttr(GRB.Attr.Lazy, model.getConstrs(), Lazy)


def simplecutloop(nn2grb, addAsCuts=False):
    model = nn2grb.model
    model.update()
    output = model.Params.OutputFlag
    model.Params.OutputFlag = 0
    Vars = model.getVars()
    VTypes = model.getAttr(GRB.Attr.VType, Vars)
    model.setAttr(GRB.Attr.VType, Vars, [GRB.CONTINUOUS]*len(Vars))

    rowsbefore = model.NumConstrs
    round = 0
    cuts = list()
    while (1):
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            break
        if output:
            print(f'Round {round}: objective value {model.ObjVal} cuts {len(nn2grb._cuts)} new {len(cuts)}')
        cuts = cut_round(nn2grb)
        if cuts:
            model.addConstrs(c for c in cuts)
            model.update()
            nn2grb._cuts += model.getConstrs()[-len(cuts):]
        else:
            break
        round += 1
    model.setAttr(GRB.Attr.VType, Vars, VTypes)
    model.update()
    if addAsCuts:
        make_lastrows_cuts(model, rowsbefore)
    model.update()
    model.Params.OutputFlag = output


def complexcutloop(model, addAsCuts=False, nodelimit=10):
    model._nn2grb = model._pipe2grb.steps[-1]
    model._mycuts = list()

    model.Params.PreCrush = 1
    model.params.NodeLimit = nodelimit
    model.optimize(ReLUcb)

    rowsbefore = model.NumConstrs
    model.addConstrs(c for c in model._mycuts)
    model.update()

    if addAsCuts:
        make_lastrows_cuts(model, rowsbefore)
    model.resetParams()
