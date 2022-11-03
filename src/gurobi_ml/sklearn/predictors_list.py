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

from .decision_tree_regressor import add_decision_tree_regressor_constr
from .gradient_boosting_regressor import add_gradient_boosting_regressor_constr
from .linear_regression import add_linear_regression_constr
from .logistic_regression import add_logistic_regression_constr
from .mlpregressor import add_mlp_regressor_constr
from .preprocessing import add_polynomial_features_constr, add_standard_scaler_constr
from .random_forest_regressor import add_random_forest_regressor_constr


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
