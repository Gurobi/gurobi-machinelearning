# Copyright © 2023-2026 Gurobi Optimization, LLC
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

"""Implements some utility tools for all scikit-learn objects."""

import numpy as np
from sklearn.utils.validation import check_is_fitted, _check_feature_names

from ..exceptions import NoSolutionError
from ..modeling import AbstractPredictorConstr


class SKgetter(AbstractPredictorConstr):
    """Utility class for sklearn regression models convertors.

    Implement some common functionalities: check predictor is fitted, output dimension, get error

    Attributes
    ----------
    predictor
        Scikit-Learn predictor embedded into Gurobi model.
    """

    def __init__(
        self,
        predictor,
        input_vars,
        **kwargs,
    ):
        check_is_fitted(predictor)
        self.predictor = predictor
        _check_feature_names(predictor, input_vars, reset=False)
        # Raises an error if prediction function is not in predictor
        if getattr(predictor, predict_function):
            pass
        self.predict_function = predict_function
        if hasattr(predictor, "n_features_in_"):
            self._input_shape = predictor.n_features_in_
        if hasattr(predictor, "n_outputs_"):
            self._output_shape = predictor.n_outputs_
        elif hasattr(predictor, "classes_"):
            self._output_shape = len(predictor.classes_)


class SKRegressor(SKgetter):
    """Utility class for sklearn regression models convertors.

    Implement some common functionalities: check predictor is fitted, output dimension, get error

    Attributes
    ----------
    predictor
        Scikit-Learn predictor embedded into Gurobi model.
    """

    def __init__(
        self,
        predictor,
        input_vars,
        **kwargs,
    ):
        SKgetter.__init__(self, predictor=predictor, input_vars=input_vars, **kwargs)

    def get_error(self, eps=None):
        """Return error in Gurobi's solution with respect to prediction from input.

        Returns
        -------
        error : ndarray of same shape as :py:attr:`gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predictor.predict(x)` and
            `y`. Where predictor is the regression this object is modeling.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution:
            X = self.input_values
            output_values = self.output_values

            predicted = self.predictor.predict(X)

            if len(predicted.shape) == 1 and len(output_values.shape) == 2:
                predicted = predicted.reshape(-1, 1)
            r_val = np.abs(predicted - output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{predicted} != {output_values}")
            return r_val

        raise NoSolution()


class SKClassifier(SKgetter):
    """Utility class for sklearn classifiers

    Implement some common functionalities: check predictor is fitted, output dimension, get error

    Attributes
    ----------
    predictor
        Scikit-Learn predictor embedded into Gurobi model.
    """

    def __init__(
        self,
        predictor,
        input_vars,
        predict_function="predict",
        **kwargs,
    ):
        # Raises an error if prediction function is not in predictor
        if predict_function == "identity" or getattr(predictor, predict_function):
            pass
        self.predict_function = predict_function
        SKgetter.__init__(self, predictor=predictor, input_vars=input_vars, **kwargs)

    def get_error(self, eps=None):
        """Return error in Gurobi's solution with respect to prediction from input.

        Returns
        -------
        error : ndarray of same shape as :py:attr:`gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predictor.predict(x)` and
            `y`. Where predictor is the regression this object is modeling.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution:
            X = self.input_values
            output_values = self.output_values

            if self.predict_function == "predict_proba":
                predict_function = self.predictor.predict_proba
            elif self.predict_function == "decision_function":
                predict_function = self.predictor.decision_function
            elif hasattr(self.predictor, "decision_function"):
                predict_function = self.predictor.decision_function
                output_values = self.linear_predictor.X
            else:
                assert hasattr(self.predictor, "predict_proba")
                predict_function = self.predictor.predict_proba
                output_values = self.linear_predictor.X
                output_values = np.exp(output_values)
                output_values /= output_values.sum(axis=1)

            predicted = predict_function(X)

            if len(predicted.shape) == 1 and len(output_values.shape) == 2:
                predicted = predicted.reshape(-1, 1)
            r_val = np.abs(predicted - output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{predicted} != {output_values}")
            return r_val

        raise NoSolutionError()


class SKtransformer(AbstractPredictorConstr):
    """Utility class for sklearn preprocessing models convertors.

    Implement some common functionalities.

    Attributes
    ----------
    transformer
        Scikit-Learn transformer embedded into Gurobi model.
    """

    def __init__(self, gp_model, transformer, input_vars, **kwargs):
        self.transformer = transformer
        if hasattr(transformer, "n_features_in_"):
            self._input_shape = transformer.n_features_in_
        if hasattr(transformer, "n_output_features_"):
            self._output_shape = transformer.n_output_features_
        check_is_fitted(transformer)
        super().__init__(gp_model, input_vars, **kwargs)

    def get_error(self, eps=None):
        """Return error in Gurobi's solution with respect to preprocessing from input.

        Returns
        -------
        error : ndarray of same shape as :py:attr:`gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `transformer.transform(x)` and
            `y`. Where transformer is the prepocessing this object is modeling.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution:
            transformer = self.transformer
            input_values = self.input_values

            transformed = transformer.transform(input_values)
            if len(transformed.shape) == 1:
                transformed = transformed.reshape(-1, 1)

            r_val = np.abs(transformed - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{transformed} != {self.output_values}")
            return r_val

        raise NoSolutionError()
