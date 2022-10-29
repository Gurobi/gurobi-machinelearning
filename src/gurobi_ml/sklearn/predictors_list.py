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

from .decisiontrees import (
    add_decision_tree_regressor_constr,
    add_gradient_boosting_regressor_constr,
    add_random_forest_regressor_constr,
)
from .mlpregressor import add_mlp_regressor_constr
from .preprocessing import add_polynomial_features_constr, add_standard_scaler_constr
from .regressions import add_linear_regression_constr, add_logistic_regression_constr

USER_PREDICTORS = {}


def register_predictor_constr(predictor, predictor_constr):
    """Register a new predictor that can be added using use_predictor_constr

    Parameters
    ----------
    predictor:
        Class of the predictor
    predictor_constr:
        Class implementing the MIP model that embeds a trained object of
        class predictor in a gurobi Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>
    """
    USER_PREDICTORS[predictor] = predictor_constr


def sklearn_transformers():
    return {
        "StandardScaler": add_standard_scaler_constr,
        "PolynomialFeatures": add_polynomial_features_constr,
    }


def sklearn_predictors():
    return {
        "LinearRegression": add_linear_regression_constr,
        "Ridge": add_linear_regression_constr,
        "Lasso": add_linear_regression_constr,
        "LogisticRegression": add_logistic_regression_constr,
        "DecisionTreeRegressor": add_decision_tree_regressor_constr,
        "GradientBoostingRegressor": add_gradient_boosting_regressor_constr,
        "RandomForestRegressor": add_random_forest_regressor_constr,
        "MLPRegressor": add_mlp_regressor_constr,
    }


def user_predictors():
    return USER_PREDICTORS
