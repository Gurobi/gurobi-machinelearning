# Copyright © 2022 Gurobi Optimization, LLC
""" Implementation for using sklearn preprocessing object in a
Guobi model"""

import gurobipy as gp

from ..exceptions import NoModel
from ..modeling import AbstractPredictorConstr, _default_name


class StandardScalerConstr(AbstractPredictorConstr):
    """Class to use a StandardScale to create scaled version of
    some Gurobi variables."""

    def __init__(self, grbmodel, scaler, input_vars, **kwargs):
        self.scaler = scaler
        super().__init__(grbmodel, input_vars, default_name=_default_name(scaler), **kwargs)

    def _create_output_vars(self, input_vars, **kwargs):
        rval = self._model.addMVar(input_vars.shape, name="scaledx")
        self._model.update()
        self._output = rval

    def mip_model(self):
        """Do the transormation on x"""
        _input = self._input
        output = self._output

        nfeat = _input.shape[1]
        scale = self.scaler.scale_
        mean = self.scaler.mean_

        output.LB = (_input.LB - mean) / scale
        output.UB = (_input.UB - mean) / scale
        self._model.addConstrs(
            (_input[:, i] - output[:, i] * scale[i] == mean[i] for i in range(nfeat)),
            name="s",
        )
        return self


class PolynomialFeaturesConstr(AbstractPredictorConstr):
    """Class to use a PolynomialFeatures to create transforms of
    some Gurobi variables."""

    def __init__(self, grbmodel, polytrans, input_vars, **kwargs):
        if polytrans.degree > 2:
            raise NoModel(polytrans, "Can only handle polynomials of degree < 2")
        self.polytrans = polytrans
        super().__init__(grbmodel, input_vars, default_name=_default_name(polytrans), **kwargs)

    def _create_output_vars(self, input_vars, **kwargs):
        out_shape = (input_vars.shape[0], self.polytrans.n_output_features_)
        rval = self._model.addMVar(out_shape, name="polyx", lb=-gp.GRB.INFINITY)
        self._model.update()
        self._output = rval

    def mip_model(self):
        """Do the transormation on x"""
        _input = self._input
        output = self._output

        nexamples, nfeat = _input.shape
        powers = self.polytrans.powers_
        assert powers.shape[0] == self.polytrans.n_output_features_
        assert powers.shape[1] == nfeat

        for k in range(nexamples):
            for i, power in enumerate(powers):
                qexpr = gp.QuadExpr()
                qexpr += 1.0
                for j, feat in enumerate(_input[k, :]):
                    if power[j] == 2:
                        qexpr *= feat.item()
                        qexpr *= feat.item()
                    elif power[j] == 1:
                        qexpr *= feat.item()
                self.model.addConstr(output[k, i] == qexpr, name=f"polyfeat[{k},{i}]")


def add_polynomial_features_constr(grbmodel, polynomial_features, input_vars, output_vars=None, **kwargs):
    return PolynomialFeaturesConstr(grbmodel, polynomial_features, input_vars, output_vars, **kwargs)


def add_standard_scaler_constr(grbmodel, scaler, input_vars, output_vars=None, **kwargs):
    return StandardScalerConstr(grbmodel, scaler, input_vars, output_vars, **kwargs)
