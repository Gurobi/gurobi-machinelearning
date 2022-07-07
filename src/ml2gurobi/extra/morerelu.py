# Copyright © 2022 Gurobi Optimization, LLC
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..activations import _name


class ReLUmin():
    ''' Model the ReLU function in a twisted way (i.e min(-x, 0)) using
        Gurobi max general constraints.'''

    def __init__(self, bigm=None, setbounds=None):
        if not setbounds:
            bigm = None
        self.bigm = bigm
        self.setbounds = setbounds

    def mip_model(self, layer):
        ''' Add weird MIP formulation for ReLU for neuron in layer'''
        if not hasattr(layer, 'mixing'):
            mixing = layer._model.addMVar(layer._output.shape, lb=-GRB.INFINITY,
                                          vtype=GRB.CONTINUOUS,
                                          name=f'mix')
            layer.mixing = mixing
            minact = layer._model.addMVar(layer._output.shape, lb=-GRB.INFINITY, ub=0.0,
                                          vtype=GRB.CONTINUOUS,
                                          name=f'minact')
            layer.minact = minact
        if layer.wmax is None:
            return
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1*self.bigm)
        if self.setbounds:
            layer._output.LB = 0.0
            layer._output.UB = np.maximum(layer.wmax, 0.0)
            layer.minact.LB = - np.maximum(layer.wmax, 0.0)
            layer.mixing.LB = -layer.wmax
            layer.mixing.UB = -layer.wmin

        input_size = layer._input.shape[1]
        for index in np.ndindex(layer._output.shape):
            k, j = index
            vact = layer._output[index]
            minact = layer.minact[index]
            mixing = sum(layer._input[k, i] * layer.coefs[i, j]
                         for i in range(input_size)) + layer.intercept[j]
            layer._model.addConstr(
                layer.mixing[index] == - mixing, name=f'mix[{index}]')
            mixing = layer.mixing[index]
            layer._model.addConstr(vact == -minact)
            layer._model.addGenConstrMin(
                minact, [mixing, 0.0], name=f'relu[{index}]')


class GRBReLU():
    ''' Model ReLU in a MIP'''

    def __init__(self, eps=1e-8, bigm=None, complement=False):
        self.eps = eps
        self.bigm = bigm
        self.complement = complement

    def mip_model(self, layer):
        ''' Add MIP formulation for ReLU for neuron in layer'''
        if not layer.zvar:
            z = layer._model.addMVar(layer._output.shape, ub=1.0, vtype=GRB.BINARY,
                                     name=f'z')
            layer.zvar = z
            mixing = layer._model.addMVar(layer._output.shape, lb=0.0, ub=-layer.wmin, vtype=GRB.CONTINUOUS,
                                          name='mix')
            layer.mixing = mixing
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1*self.bigm)
        layer._output.LB = 0.0
        layer._output.UB = np.maximum(layer.wmax, 0.0)
        layer.mixing.LB = 0.0
        layer.mixing.UB = np.maximum(-layer.wmin, 0.0)

        input_size = layer._input.shape[1]
        for index in np.ndindex(layer._output.shape):
            k, j = index
            lb = layer.wmin[index]
            ub = layer.wmax[index]
            vact = layer._output[index]
            vmix = layer.mixing[index]
            assert ub >= lb

            mixing = sum(layer._input[k, i] * layer.coefs[i, j]
                         for i in range(input_size)) + layer.intercept[j]
            if ub > self.eps and lb < -self.eps:
                layer._model.addConstr(
                    vact - vmix == mixing, name=f'mix[{index}]')
                mixing = layer.mixing[index]
                vz = layer.zvar[index]
                if self.complement or -lb < ub:
                    vz = 1 - vz
                layer._model.addConstr(
                    vmix <= -lb*(1-vz), name=f'low[{index}]')
                layer._model.addConstr(vact <= ub*vz, name=f'vub[{index}]')
            elif ub <= self.eps:
                vact.UB = 0.0
                vact.LB = 0.0
                vmix.UB = 0.0
            else:
                assert lb >= -self.eps
                layer._model.addConstr(vact == mixing, name=f'mix[{index}]')

    @staticmethod
    def reset_bounds(layer):
        '''Reset the bounds in layer corresponding to modeling ReLU'''
        layer.zvar.UB = 1.0
        layer.zvar.LB = 0.0
        layer._output.LB = 0.0
        layer._output.UB = np.maximum(layer.wmax, 0.0)


class ReLUM():
    ''' Model ReLU in a MIP'''

    def __init__(self, eps=1e-8, bigm=None, expand=False):
        self.eps = eps
        self.bigm = bigm
        self.expand = expand

    def mip_model(self, layer):
        ''' Add MIP formulation for ReLU for neuron in layer'''
        if not layer.zvar:
            z = layer._model.addMVar(layer._output.shape, ub=1.0, vtype=GRB.BINARY,
                                     name=f'z')
            layer.zvar = z
            if not self.expand:
                mixing = layer._model.addMVar(layer._output.shape, lb=layer.wmin, ub=layer.wmax, vtype=GRB.CONTINUOUS,
                                              name='mix')
                layer.mixing = mixing
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1*self.bigm)
        layer._output.LB = 0.0
        layer._output.UB = np.maximum(layer.wmax, 0.0)
        if not self.expand:
            layer.mixing.LB = layer.wmin
            layer.mixing.UB = layer.wmax

        input_size = layer._input.shape[1]
        for index in np.ndindex(layer._output.shape):
            k, j = index
            lb = layer.wmin[index]
            ub = layer.wmax[index]
            vact = layer._output[index]
            assert ub >= lb
            constrname = _name(index)
            mixing = sum(layer._input[k, i] * layer.coefs[i, j]
                         for i in range(input_size)) + layer.intercept[j]
            if ub > self.eps and lb < -self.eps:
                if not self.expand:
                    layer._model.addConstr(
                        layer.mixing[index] == mixing, name=_name(index, 'mix'))
                    mixing = layer.mixing[index]
                vz = layer.zvar[index]
                layer._model.addConstr(
                    vact >= mixing, name=_name(index, 'low'))
                layer._model.addConstr(
                    vact <= ub*vz, name=_name(index, 'vub1'))
                layer._model.addConstr(vact <= mixing - lb * (1 - vz),
                                       name=_name(index, 'vub2'))
            elif ub <= self.eps:
                vact.UB = 0.0
                vact.LB = 0.0
            else:
                assert lb >= -self.eps
                layer._model.addConstr(vact == mixing, name=constrname)

    @staticmethod
    def forward(input_values):
        '''Return ReLU of input_values'''
        return np.maximum(0.0, input_values)

    @staticmethod
    def forward_fixing(layer, input_values, threshold=-20):
        '''Fix binaries according to input_values'''
        zvar = layer.zvar
        if threshold < 0:
            threshold = -int(threshold)
            threshold = np.sort(np.abs(input_values))[0, threshold]
        zvar[input_values < 0.0].UB = 0
        zvar[input_values >= 0.0].LB = 1

        closetozero = zvar[np.abs(input_values) <= threshold].tolist()
        return closetozero

    @staticmethod
    def reset_bounds(layer):
        '''Reset the bounds in layer corresponding to modeling ReLU'''
        layer.zvar.UB = 1.0
        layer.zvar.LB = 0.0
        layer._output.LB = 0.0
        layer._output.UB = np.maximum(layer.wmax, 0.0)


class reluOBBT():
    def __init__(self, obbt_rel):
        assert obbt_rel in ('either', 'comb', 'both')
        self.obbt_rel = obbt_rel

    def mip_model(self, layer):
        ''' This is the convex hull of ReLU without binary (just for doing OBBT)'''
        if layer.wmax is not None:
            layer._output.LB = 0.0
            layer._output.UB = np.maximum(layer.wmax, 0.0)

        input_size = layer._input.shape[1]
        for index in np.ndindex(layer._output.shape):
            k, j = index
            lb = layer.wmin[index]
            ub = layer.wmax[index]
            vact = layer._output[index]

            constrname = f'[{index}]'.replace(' ', '')
            mixing = sum(layer._input[k, i] * layer.coefs[i, j]
                         for i in range(input_size)) + layer.intercept[j]
            if ub < 1e-8:
                layer._model.addConstr(vact <= 0, name=constrname+'_inactive')
                return
            elif lb > -1e-8:
                layer._model.addConstr(
                    mixing == vact, name=constrname+'_active')
                return

            alpha = ub/(ub - lb)

            if self.obbt_rel == 'comb':
                layer._model.addConstr(
                    vact >= alpha*mixing, name=constrname+'_low')
            elif self.obbt_rel == 'either':
                if abs(ub) > abs(lb):
                    layer._model.addConstr(
                        vact >= mixing, name=constrname+'_low')
                else:
                    layer._model.addConstr(vact >= 0, name=constrname+'_low')
            else:
                layer._model.addConstr(vact >= mixing, name=constrname+'_low1')
                layer._model.addConstr(vact >= 0, name=constrname+'_low2')

            layer._model.addConstr(
                vact <= alpha*mixing - lb*alpha, name=constrname+'_up')


class reluPart(ReLUM):
    def __init__(self, n_partitions=2):
        super().__init__()
        self.n_partitions = n_partitions

    @staticmethod
    def _sigma_minmax(X, w, partitions):
        N = len(partitions)
        part_shape = (N, X.shape[0], w.shape[1])
        wpos = np.maximum(w, 0.0)
        wneg = np.minimum(w, 0.0)
        sigma_min = np.empty(part_shape)
        sigma_max = np.empty(part_shape)
        for p in range(N):
            for i in range(w.shape[1]):
                part = partitions[p][:, i]
                for k in range(X.shape[0]):
                    sigma_min[p, k, i] = X[k, part].LB @ wpos[part,
                                                              i] + X[k, part].UB @ wneg[part, i]
                    sigma_max[p, k, i] = X[k, part].UB @ wpos[part,
                                                              i] + X[k, part].LB @ wneg[part, i]
        return (sigma_min, sigma_max)

    def conv(self, layer, index):
        k, j = index
        model = layer._model
        invar = layer._input[k, :]
        vact = layer._output[index]
        w = layer.coefs[:, j]
        w0 = layer.intercept[j]
        lb = layer.sigma_min[:, k, j]
        ub = layer.sigma_max[:, k, j]
        vsigma = layer.sigma[:, k, j]
        vz = layer.zvar[k, j]

        mixing = layer.getmixing(index)
        name = layer.getname(index, 'relupart')

        for p in range(self.n_partitions):
            part = layer.partitions[p][:, j]
            s = vsigma[p]
            part_expr = sum(invar[p] * w[p] for p in part)
            c = model.addConstr(s >= (1 - vz) * lb[p], name=name+f'_{p}_1')
            layer.constrs.append(c)
            c = model.addConstr(s <= (1 - vz) * ub[p], name=name+f'_{p}_2')
            layer.constrs.append(c)
            c = model.addConstr(part_expr - s >= vz *
                                lb[p], name=name+f'_{p}_3')
            layer.constrs.append(c)
            c = model.addConstr(part_expr - s <= vz *
                                ub[p], name=name+f'_{p}_4')
            layer.constrs.append(c)
        c = model.addConstr(mixing - w0 - sum(vsigma[:].tolist()) + vz * w0 <= 0,
                            name=name+'_link1')
        layer.constrs.append(c)
        c = model.addConstr(gp.quicksum(vsigma[:].tolist()) + (1 - vz) * w0 >= 0,
                            name=name+'_link2')
        layer.constrs.append(c)
        c = model.addConstr(vact == gp.quicksum(vsigma[:].tolist()) + (1 - vz) * w0,
                            name=name+'_link3')
        layer.constrs.append(c)

    def preprocess(self, layer):
        invar = layer._input
        coefs = layer.coefs
        ReLUM.preprocess(self, layer)
        if "partitions" not in layer.__dict__:
            model = layer._model
            part_shape = (self.n_partitions, invar.shape[0], coefs.shape[1])
            partitions = np.array_split(np.argsort(coefs, axis=0), 2, axis=0)
            sigma = model.addMVar(part_shape, lb=-GRB.INFINITY,
                                  name=f'sigma')
            sigma_min, sigma_max = self._sigma_minmax(invar, coefs, partitions)
            sigma.LB = np.minimum(sigma_min, 0)
            sigma.UB = np.maximum(sigma_max, 0)
            model.update()
            layer.partitions = partitions
            layer.sigma = sigma
            layer.sigma_min = sigma_min
            layer.sigma_max = sigma_max
        else:
            sigma = layer.sigma
            sigma_min, sigma_max = self._sigma_minmax(
                invar, coefs, layer.partitions)
            assert layer.sigma_max is not None
            assert layer.sigma_min is not None
            sigma_max = np.minimum(sigma_max, layer.sigma_max)
            sigma_min = np.maximum(sigma_min, layer.sigma_min)
            layer.sigma_min = sigma_min
            layer.sigma_max = sigma_max
            sigma.LB = np.minimum(sigma_min, 0)
            sigma.UB = np.maximum(sigma_max, 0)
            layer._model.update()


def obbt_part(layer, round):
    m = layer._model
    obj = m.getObjective()
    objsense = m.ModelSense
    output = m.Params.OutputFlag
    savemdethod = m.Params.Method
    input_vars = layer._input
    layer_coefs = layer.coefs
    layer_intercept = layer.intercept

    VTypes = m.getAttr(GRB.Attr.VType, m.getVars())

    m.Params.OutputFlag = 0
    m.setAttr(GRB.Attr.VType, m.getVars(), GRB.CONTINUOUS)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    print(f'Round {round} objval {m.ObjVal}')

    m.Params.Method = 0

    n_part = layer.sigma.shape[0]
    n = layer.sigma.shape[1]
    layer_size = layer.sigma.shape[2]

    sigma_min = layer.sigma_min
    sigma_max = layer.sigma_max
    partitions = layer.partitions
    eps = 1e-7
    n_strengthened = 0
    n_fixed = 0
    done = 0
    alreadfixed = (layer.wmin > -eps) | (layer.wmax < eps)
    for j in range(layer_size):

        w = layer_coefs[:, j]
        for p in range(n_part):
            part = partitions[p][:, j]
            for k in range(n):
                if alreadfixed[k, j]:
                    continue
                done += 1
                sum(input_vars[k, p] * w[p] for p in part)
                m.setObjective(sum(input_vars[k, p] * w[p] for p in part),
                               GRB.MAXIMIZE)
                m.optimize()
                assert m.Status == GRB.OPTIMAL
                if m.ObjVal < sigma_max[p, k, j] - 1e-5:
                    sigma_max[p, k, j] = m.ObjVal + 1e-5
                    n_strengthened += 1
    print(
        f'OBBT strengthened {n_strengthened} and fixed {n_fixed} upper bounds on layer (did {done})')

    layer.wmax = np.minimum(layer.sigma_max.sum(
        axis=0) + layer_intercept, layer.wmax)

    total_strengthened = n_strengthened
    n_strengthened = 0
    n_fixed = 0
    done = 0
    alreadfixed = (layer.wmin > -eps) | (layer.wmax < eps)
    for j in range(layer_size):
        w = layer_coefs[:, j]
        for p in range(n_part):
            part = partitions[p][:, j]
            for k in range(n):
                if alreadfixed[k, j]:
                    continue
                done += 1
                m.setObjective(sum(input_vars[k, p] * w[p] for p in part),
                               GRB.MINIMIZE)
                m.optimize()
                assert m.Status == GRB.OPTIMAL
                if m.ObjVal > sigma_min[p, k, j] + 1e-5:
                    sigma_min[p, k, j] = m.ObjVal - 1e-5
                    n_strengthened += 1

    print(
        f'OBBT strengthened {n_strengthened} and fixed {n_fixed} lower bounds on layer (did {done})')

    layer.wmin = np.maximum(layer.sigma_min.sum(
        axis=0) + layer_intercept, layer.wmin)
    layer.wmax = np.minimum(layer.sigma_max.sum(
        axis=0) + layer_intercept, layer.wmax)
    # Restore model
    m.Params.Method = savemdethod
    m.Params.OutputFlag = output
    m.setAttr(GRB.Attr.VType, m.getVars(), VTypes)
    m.setObjective(obj)
    m.ModelSense = objsense
    m.update()
    return total_strengthened + n_strengthened
