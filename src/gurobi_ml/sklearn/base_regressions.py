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

""" Module for inserting simple Scikit-Learn regression models into a gurobipy model

All linear models should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`

"""


from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


class BaseSKlearnRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        output_type="",
        **kwargs,
    ):
        self.n_outputs_ = 1
        SKgetter.__init__(self, predictor, output_type, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            gp_model,
            input_vars,
            output_vars,
            **kwargs,
        )

    def add_regression_constr(self):
        """Add the prediction constraints to Gurobi"""
        coefs = self.predictor.coef_.reshape(-1, 1)
        intercept = self.predictor.intercept_
        self.gp_model.addConstr(self.output == self.input @ coefs + intercept, name="linreg")
