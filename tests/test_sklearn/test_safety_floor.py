import unittest
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.tree import DecisionTreeRegressor
from gurobi_ml.sklearn import add_decision_tree_regressor_constr


class TestSafetyFloor(unittest.TestCase):
    def test_safety_floor(self):
        # Create a simple decision tree with a threshold close to zero
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        # Force a split at 0.00001
        dt = DecisionTreeRegressor(max_depth=1)
        dt.fit(X, y)
        # Manually set the threshold to a small value
        dt.tree_.threshold[0] = 0.00001

        # Test without safety_floor (threshold should be 0.00001)
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_var = m.addMVar((1, 1), lb=0, ub=1)
            # Use a large epsilon so we can see the effect
            pred = add_decision_tree_regressor_constr(m, dt, x_var, epsilon=0.0)

            # The split is x <= 0.00001
            # If x = 0.000005, it should go to left leaf
            x_var.setAttr(GRB.Attr.LB, 0.000005)
            x_var.setAttr(GRB.Attr.UB, 0.000005)
            m.optimize()
            self.assertEqual(m.Status, GRB.OPTIMAL)

        # Test with safety_floor = 0.01 (threshold should be clamped to 0.01)
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_var = m.addMVar((1, 1), lb=0, ub=1)
            pred = add_decision_tree_regressor_constr(
                m, dt, x_var, epsilon=0.0, safety_floor=0.01
            )

            # Now the split should be x <= 0.01
            # If x = 0.005, it should still go to left leaf (which it would anyway)
            # But let's verify if we can force it to be at least safety_floor

            # If we fix x = 0.005, and safety_floor is 0.01, the threshold is 0.01.
            # So x <= 0.01 is true.

            # If we fix x = 0.00001, it is <= 0.00001.
            # If we fix x = 0.005, it is > 0.00001 but <= 0.01.

            # Without safety_floor, x = 0.005 would go to RIGHT leaf.
            # With safety_floor = 0.01, x = 0.005 should go to LEFT leaf.

            x_var.setAttr(GRB.Attr.LB, 0.005)
            x_var.setAttr(GRB.Attr.UB, 0.005)
            m.optimize()
            self.assertEqual(m.Status, GRB.OPTIMAL)
            # Left leaf value is y[0] = 0.0, right leaf value is y[1] = 1.0
            self.assertAlmostEqual(pred.output.X[0, 0], 0.0)

        # Verify without safety_floor it goes to right leaf
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_var = m.addMVar((1, 1), lb=0, ub=1)
            pred = add_decision_tree_regressor_constr(m, dt, x_var, epsilon=0.0)
            x_var.setAttr(GRB.Attr.LB, 0.005)
            x_var.setAttr(GRB.Attr.UB, 0.005)
            m.optimize()
            self.assertEqual(m.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred.output.X[0, 0], 1.0)

    def test_safety_floor_negative(self):
        # Create a simple decision tree with a threshold close to zero
        X = np.array([[-1.0], [0.0]])
        y = np.array([0.0, 1.0])
        dt = DecisionTreeRegressor(max_depth=1)
        dt.fit(X, y)
        # Manually set the threshold to a small negative value
        dt.tree_.threshold[0] = -0.00001

        # Test with safety_floor = 0.01 (threshold should be clamped to -0.01)
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_var = m.addMVar((1, 1), lb=-1, ub=0)
            pred = add_decision_tree_regressor_constr(
                m, dt, x_var, epsilon=0.0, safety_floor=0.01
            )

            # Threshold -0.00001 -> abs is 0.00001 < 0.01 -> clamped to sign(-0.00001) * 0.01 = -0.01
            # Split is x <= -0.01

            # If x = -0.005, it is > -0.01, so it should go to RIGHT leaf (value 1.0)
            x_var.setAttr(GRB.Attr.LB, -0.005)
            x_var.setAttr(GRB.Attr.UB, -0.005)
            m.optimize()
            self.assertEqual(m.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred.output.X[0, 0], 1.0)

        # Without safety_floor, x = -0.005 is > -0.00001? No, -0.005 < -0.00001.
        # So x = -0.005 goes to LEFT leaf (value 0.0)
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_var = m.addMVar((1, 1), lb=-1, ub=0)
            pred = add_decision_tree_regressor_constr(m, dt, x_var, epsilon=0.0)
            x_var.setAttr(GRB.Attr.LB, -0.005)
            x_var.setAttr(GRB.Attr.UB, -0.005)
            m.optimize()
            self.assertEqual(m.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred.output.X[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
