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


"""Module for user to register a predictor for the function add_predictor_constr."""
USER_PREDICTORS = {}


def register_predictor_constr(predictor, predictor_constr):
    """Register a new predictor that can be added using use_predictor_constr.

    Parameters
    ----------
    predictor:
        Class of the predictor
    predictor_constr:
        Class implementing the MIP formulation of a trained object of class predictor
        in a gurobi :gurobipy:`Model`.
    """
    USER_PREDICTORS[predictor] = predictor_constr


def user_predictors():
    """Returns dictionary of user defined predictors."""
    return USER_PREDICTORS
