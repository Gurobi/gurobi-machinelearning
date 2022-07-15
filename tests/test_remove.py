""" Test for adding and removing predictor objects"""
import unittest

import gurobipy as gp
from base_cases import DiabetesCases

from ml2gurobi import add_predictor_constr


class TestAddRemove(unittest.TestCase):
    """Test Adding and Removing submodels and check counts."""

    def check_counts(self, model, reg2gurobi, numvars):
        """Assert counts are ok"""
        self.assertEqual(model.NumVars, numvars + len(reg2gurobi.vars))
        self.assertEqual(model.NumSOS, len(reg2gurobi.sos))
        self.assertEqual(model.NumConstrs, len(reg2gurobi.constrs))
        self.assertEqual(model.NumQConstrs, len(reg2gurobi.qconstrs))
        self.assertEqual(model.NumGenConstrs, len(reg2gurobi.genconstrs))

    def add_remove(self, predictor, input_shape, output_shape, nonconvex):
        """Add and remove the predictor to model"""
        with gp.Model() as model:
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            pred2grb = add_predictor_constr(model, predictor, x, y)

            self.check_counts(model, pred2grb, numvars)

            pred2grb.remove()
            model.update()
            self.check_counts(model, pred2grb, numvars)
            self.assertEqual(model.NumConstrs, 0)
            self.assertEqual(model.NumGenConstrs, 0)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertEqual(model.NumVars, numvars)

    def add_remove_no_output(self, predictor, input_shape, output_shape, nonconvex):
        """Add and remove the predictor to model"""
        with gp.Model() as model:
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            pred2grb = add_predictor_constr(model, predictor, x)

            self.assertEqual(pred2grb.output.shape[0], output_shape[0])

            self.check_counts(model, pred2grb, numvars)

            pred2grb.remove()
            model.update()
            self.check_counts(model, pred2grb, numvars)
            self.assertEqual(model.NumConstrs, 0)
            self.assertEqual(model.NumGenConstrs, 0)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertEqual(model.NumVars, numvars)

    def test_diabetes_with_outputvar(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove(**onecase)

    def test_diabetes_without_outputvar(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove_no_output(**onecase)
