# Copyright © 2022 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Implementation for using sklearn preprocessing object in a
Guobi model"""

import gurobipy as gp

from ..exceptions import NoModel
from ..modeling import AbstractPredictorConstr, _default_name


def add_polynomial_features_constr(grbmodel, polynomial_features, input_vars, **kwargs):
    """Use `polynomial_features` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    polynomial_features: :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`
        The polynomial features to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    sklearn.preprocessing.PolynomialFeaturesConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return PolynomialFeaturesConstr(grbmodel, polynomial_features, input_vars, **kwargs)


def add_standard_scaler_constr(grbmodel, standard_scaler, input_vars, **kwargs):
    """Use a `standard_scaler` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    standard_scaler: :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler`
        The standard scaler to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    sklearn.preprocessing.StandardScalerConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return StandardScalerConstr(grbmodel, standard_scaler, input_vars, **kwargs)


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

    def _mip_model(self):
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

    def _mip_model(self):
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
