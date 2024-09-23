# Copyright © 2023 Gurobi Optimization, LLC
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

import gurobipy as gp
import numpy as np

from .base_predictor_constr import AbstractPredictorConstr

try:
    from gurobipy import nlfunc

    _HAS_NL_EXPR = True
except ImportError:
    _HAS_NL_EXPR = False

_SKIP_VARS = False


def argmax(
    predictor_model: AbstractPredictorConstr, linear_predictor: gp.MLinExpr, **kwargs
):
    gp_model: gp.Model = predictor_model.gp_model
    output: gp.MVar = predictor_model.output

    if "epsilon" in kwargs:
        epsilon = kwargs["epsilon"]
    else:
        epsilon = 0.0

    linear_predictor_vars = gp_model.addMVar(
        output.shape, lb=-gp.GRB.INFINITY, name="linear_predictor"
    )
    gp_model.addConstr(linear_predictor_vars == linear_predictor, name="linreg")
    predictor_model.linear_predictor = linear_predictor_vars

    gp_model.addConstr(output.sum(axis=1) == 1)

    # Do the argmax
    # We use indicators (a lot of them)
    for index in np.ndindex(output.shape):
        i, j = index
        for k in np.ndindex(output.shape[1]):
            if k == j:
                continue
            gp_model.addGenConstrIndicator(
                output[index],
                1,
                linear_predictor_vars[index] - linear_predictor_vars[i, k],
                gp.GRB.GREATER_EQUAL,
                epsilon,
            )


def softmax(
    predictor_model: AbstractPredictorConstr, linear_predictor: gp.MLinExpr, **kwargs
):
    """Add the prediction constraints to Gurobi."""
    gp_model: gp.Model = predictor_model.gp_model
    output: gp.MVar = predictor_model.output
    try:
        predict_function: str = predictor_model.predict_function
    except AttributeError:
        predict_function = "predict_proba"

    if "epsilon" in kwargs:
        epsilon = kwargs["epsilon"]
    else:
        epsilon = 0.0

    num_gc = gp_model.NumGenConstrs

    gp_model.update()

    if _HAS_NL_EXPR:
        # We want to write y_j = e^z_j / sum_j=1^k e^z_j
        # y_j are the output variable z_j = input @ coefs + intercepts

        # Store the e^z_j in a nonlinear expression
        if _SKIP_VARS:
            exponentials = nlfunc.exp(linear_predictor)
        else:
            exponentials = gp_model.addMVar(output.shape, name="exponentials")
            gp_model.addConstr(exponentials == nlfunc.exp(linear_predictor))
        # The denominator is the sum over the first axis
        denominator = exponentials.sum(axis=1)

        # Voila!
        gp_model.addConstr(output == exponentials / denominator, name=f"multlog")
    else:
        # How boy that is tedious you don't want not to use Gurobi 12!
        linear_predictor_vars = gp_model.addMVar(
            output.shape, lb=-gp.GRB.INFINITY, name="linear_predictor"
        )
        gp_model.addConstr(linear_predictor_vars == linear_predictor, name="linreg")
        predictor_model.linear_predictor = linear_predictor_vars

        exponentials = gp_model.addMVar(output.shape)
        exponentials = exponentials
        denominator = gp_model.addMVar((output.shape[0]), lb=epsilon)

        num_gc = gp_model.NumGenConstrs

        for index in np.ndindex(output.shape):
            gp_model.addGenConstrExp(
                linear_predictor_vars[index],
                exponentials[index],
                name=predictor_model._indexed_name(index, "exponential"),
            )
        gp_model.update()
        try:
            attributes = predictor_model.attributes
        except AttributeError:
            attributes = {}
        for gen_constr in gp_model.getGenConstrs()[num_gc:]:
            for attr, val in attributes.items():
                gen_constr.setAttr(attr, val)
        gp_model.addConstr(denominator == exponentials.sum(axis=1))

        gp_model.addConstrs(
            output[i, :] * denominator[i] == exponentials[i, :]
            for i in range(output.shape[0])
        )
    gp_model.addConstr(output <= 1)
    gp_model.addConstr(output >= 0)
