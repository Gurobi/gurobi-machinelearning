import unittest
import numpy as np
import pandas as pd
import gurobipy as gp
from sklearn.linear_model import LinearRegression
from gurobi_ml import add_predictor_constr


class TestPandasSeries(unittest.TestCase):
    def test_series_input(self):
        # Create data
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = np.array([10, 20, 30])

        # Train model
        reg = LinearRegression()
        reg.fit(X, y)

        # Gurobi model
        with gp.Env() as env, gp.Model(env=env) as model:
            # Input variables as a Series
            x_vars = pd.Series(
                model.addVars(X.columns, name="x"), index=X.columns, name="sample1"
            )

            # Add predictor constraint
            pred_constr = add_predictor_constr(model, reg, x_vars)

            # Set values for x_vars
            for col in X.columns:
                model.addConstr(x_vars[col] == X.loc[0, col])

            model.optimize()

            # Check solution values
            self.assertIsInstance(pred_constr.input_values, pd.DataFrame)
            self.assertEqual(pred_constr.input_values.shape, (1, 2))

            # Check error
            error = pred_constr.get_error()
            self.assertLess(np.max(error), 1e-6)

    def test_series_output(self):
        # Test Case 2: Many samples, one output as Series
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = np.array([10, 20, 30])

        reg = LinearRegression()
        reg.fit(X, y)

        with gp.Env() as env, gp.Model(env=env) as model:
            # Input as DataFrame (3 samples)
            x_vars = pd.DataFrame(
                [
                    [model.addVar(name=f"x_{i}_{j}") for j in X.columns]
                    for i in range(len(X))
                ],
                columns=X.columns,
                index=X.index,
            )

            # Output as Series (3 samples)
            y_vars = pd.Series(
                model.addVars(X.index, name="y"), index=X.index, name="output"
            )

            # Add predictor constraint
            pred_constr = add_predictor_constr(model, reg, x_vars, y_vars)

            # Set values
            for i in range(len(X)):
                for col in X.columns:
                    model.addConstr(x_vars.loc[i, col] == X.loc[i, col])

            model.optimize()

            # Check shapes
            self.assertEqual(pred_constr.output.shape, (3, 1))
            self.assertIsInstance(pred_constr.output_values, np.ndarray)
            self.assertEqual(pred_constr.output_values.shape, (3, 1))

            error = pred_constr.get_error()
            self.assertLess(np.max(error), 1e-6)


if __name__ == "__main__":
    unittest.main()
