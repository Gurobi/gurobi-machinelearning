# Copyright © 2022 Gurobi Optimization, LLC
import numpy as np
from gurobipy import GRB

from ..activations import _name


class ReLUmin:
    """Model the ReLU function in a twisted way (i.e min(-x, 0)) using
    Gurobi max general constraints."""

    def __init__(self, bigm=None, setbounds=None):
        if not setbounds:
            bigm = None
        self.bigm = bigm
        self.setbounds = setbounds

    def mip_model(self, layer):
        """Add weird MIP formulation for ReLU for neuron in layer"""
        if not hasattr(layer, "mixing"):
            mixing = layer.model.addMVar(layer._output.shape, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mix")
            layer.mixing = mixing
            minact = layer.model.addMVar(
                layer._output.shape,
                lb=-GRB.INFINITY,
                ub=0.0,
                vtype=GRB.CONTINUOUS,
                name="minact",
            )
            layer.minact = minact
        if layer.wmax is None:
            return
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1 * self.bigm)
        if self.setbounds:
            layer._output.LB = 0.0
            layer._output.UB = np.maximum(layer.wmax, 0.0)
            layer.minact.LB = -np.maximum(layer.wmax, 0.0)
            layer.mixing.LB = -layer.wmax
            layer.mixing.UB = -layer.wmin

        input_size = layer._input.shape[1]
        for index in np.ndindex(layer._output.shape):
            k, j = index
            vact = layer._output[index]
            minact = layer.minact[index]
            mixing = sum(layer._input[k, i] * layer.coefs[i, j] for i in range(input_size)) + layer.intercept[j]
            layer.model.addConstr(layer.mixing[index] == -mixing, name=_name(index, "mix"))
            mixing = layer.mixing[index]
            layer.model.addConstr(vact == -minact)
            layer.model.addGenConstrMin(minact, [mixing, 0.0], name=_name(index, "relu"))


class GRBReLU:
    """Model ReLU in a MIP"""

    def __init__(self, eps=1e-8, bigm=None, complement=False):
        self.eps = eps
        self.bigm = bigm
        self.complement = complement

    def mip_model(self, layer):
        """Add MIP formulation for ReLU for neuron in layer"""
        if not layer.zvar:
            z = layer.model.addMVar(layer._output.shape, ub=1.0, vtype=GRB.BINARY, name="z")
            layer.zvar = z
            mixing = layer.model.addMVar(
                layer._output.shape,
                lb=0.0,
                ub=-layer.wmin,
                vtype=GRB.CONTINUOUS,
                name="mix",
            )
            layer.mixing = mixing
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1 * self.bigm)
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

            mixing = sum(layer._input[k, i] * layer.coefs[i, j] for i in range(input_size)) + layer.intercept[j]
            if ub > self.eps and lb < -self.eps:
                layer.model.addConstr(vact - vmix == mixing, name=_name(index, "mix"))
                mixing = layer.mixing[index]
                vz = layer.zvar[index]
                if self.complement or -lb < ub:
                    vz = 1 - vz
                layer.model.addConstr(vmix <= -lb * (1 - vz), name=_name(index, "low"))
                layer.model.addConstr(vact <= ub * vz, name=_name(index, "vub"))
            elif ub <= self.eps:
                vact.UB = 0.0
                vact.LB = 0.0
                vmix.UB = 0.0
            else:
                assert lb >= -self.eps
                layer.model.addConstr(vact == mixing, name=_name(index, "mix"))

    @staticmethod
    def reset_bounds(layer):
        """Reset the bounds in layer corresponding to modeling ReLU"""
        layer.zvar.UB = 1.0
        layer.zvar.LB = 0.0
        layer._output.LB = 0.0
        layer._output.UB = np.maximum(layer.wmax, 0.0)


class ReLUM:
    """Model ReLU in a MIP"""

    def __init__(self, eps=1e-8, bigm=None, expand=False):
        self.eps = eps
        self.bigm = bigm
        self.expand = expand

    def mip_model(self, layer):
        """Add MIP formulation for ReLU for neuron in layer"""
        if not layer.zvar:
            z = layer.model.addMVar(layer._output.shape, ub=1.0, vtype=GRB.BINARY, name="z")
            layer.zvar = z
            if not self.expand:
                mixing = layer.model.addMVar(
                    layer._output.shape,
                    lb=layer.wmin,
                    ub=layer.wmax,
                    vtype=GRB.CONTINUOUS,
                    name="mix",
                )
                layer.mixing = mixing
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1 * self.bigm)
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
            mixing = sum(layer._input[k, i] * layer.coefs[i, j] for i in range(input_size)) + layer.intercept[j]
            if ub > self.eps and lb < -self.eps:
                if not self.expand:
                    layer.model.addConstr(layer.mixing[index] == mixing, name=_name(index, "mix"))
                    mixing = layer.mixing[index]
                vz = layer.zvar[index]
                layer.model.addConstr(vact >= mixing, name=_name(index, "low"))
                layer.model.addConstr(vact <= ub * vz, name=_name(index, "vub1"))
                layer.model.addConstr(vact <= mixing - lb * (1 - vz), name=_name(index, "vub2"))
            elif ub <= self.eps:
                vact.UB = 0.0
                vact.LB = 0.0
            else:
                assert lb >= -self.eps
                layer.model.addConstr(vact == mixing, name=_name(index, "mix"))

    @staticmethod
    def forward(input_values):
        """Return ReLU of input_values"""
        return np.maximum(0.0, input_values)

    @staticmethod
    def forward_fixing(layer, input_values, threshold=-20):
        """Fix binaries according to input_values"""
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
        """Reset the bounds in layer corresponding to modeling ReLU"""
        layer.zvar.UB = 1.0
        layer.zvar.LB = 0.0
        layer._output.LB = 0.0
        layer._output.UB = np.maximum(layer.wmax, 0.0)


class reluOBBT:
    def __init__(self, obbt_rel):
        assert obbt_rel in ("either", "comb", "both")
        self.obbt_rel = obbt_rel

    def mip_model(self, layer):
        """This is the convex hull of ReLU without binary (just for doing OBBT)"""
        if layer.wmax is not None:
            layer._output.LB = 0.0
            layer._output.UB = np.maximum(layer.wmax, 0.0)

        input_size = layer._input.shape[1]
        for index in np.ndindex(layer._output.shape):
            k, j = index
            lb = layer.wmin[index]
            ub = layer.wmax[index]
            vact = layer._output[index]

            constrname = f"[{index}]".replace(" ", "")
            mixing = sum(layer._input[k, i] * layer.coefs[i, j] for i in range(input_size)) + layer.intercept[j]
            if ub < 1e-8:
                layer.model.addConstr(vact <= 0, name=constrname + "_inactive")
                return
            elif lb > -1e-8:
                layer.model.addConstr(mixing == vact, name=constrname + "_active")
                return

            alpha = ub / (ub - lb)

            if self.obbt_rel == "comb":
                layer.model.addConstr(vact >= alpha * mixing, name=constrname + "_low")
            elif self.obbt_rel == "either":
                if abs(ub) > abs(lb):
                    layer.model.addConstr(vact >= mixing, name=constrname + "_low")
                else:
                    layer.model.addConstr(vact >= 0, name=constrname + "_low")
            else:
                layer.model.addConstr(vact >= mixing, name=constrname + "_low1")
                layer.model.addConstr(vact >= 0, name=constrname + "_low2")

            layer.model.addConstr(vact <= alpha * mixing - lb * alpha, name=constrname + "_up")
