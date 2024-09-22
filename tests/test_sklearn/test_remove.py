"""Test for adding and removing predictor objects"""

import random
import unittest

import gurobipy as gp

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoSolutionError

from .sklearn_cases import DiabetesCases, IrisBinaryCases


class TestAddRemove(unittest.TestCase):
    """Test adding and removing predictor constraints and check counts."""

    def check_counts(self, gp_model, pred_constr, numvars):
        """Assert counts are ok"""
        self.assertEqual(gp_model.NumVars, numvars + len(pred_constr.vars))
        self.assertEqual(gp_model.NumSOS, len(pred_constr.sos))
        self.assertEqual(gp_model.NumConstrs, len(pred_constr.constrs))
        self.assertEqual(gp_model.NumQConstrs, len(pred_constr.qconstrs))
        self.assertEqual(gp_model.NumGenConstrs, len(pred_constr.genconstrs))

    @staticmethod
    def truncate_shapes(input_shape, output_shape, maxexamples=5):
        assert len(input_shape) == 2
        nexamples = min(input_shape[0], maxexamples)
        if len(output_shape) == 2:
            assert output_shape[0] == input_shape[0]
            output_shape = (nexamples, output_shape[1])
        else:
            assert len(output_shape) == 1
            output_shape = (nexamples,)
        input_shape = (nexamples, input_shape[1])
        return (input_shape, output_shape)

    def add_remove(self, predictor, input_shape, output_shape):
        """Add and remove the predictor to model using MVar"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as gp_model:
            x = gp_model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            gp_model.update()
            numvars = gp_model.numvars

            gp_model.Params.OutputFlag = 0
            pred_constr = add_predictor_constr(gp_model, predictor, x, y)

            self.check_counts(gp_model, pred_constr, numvars)

            with self.assertRaises(NoSolutionError):
                pred_constr.get_error()

            pred_constr.remove()
            gp_model.update()
            self.check_counts(gp_model, pred_constr, numvars)
            self.assertEqual(gp_model.NumConstrs, 0)
            self.assertEqual(gp_model.NumGenConstrs, 0)
            self.assertEqual(gp_model.NumQConstrs, 0)
            self.assertEqual(gp_model.NumVars, numvars)

    def add_remove_wrong_input(self, predictor, input_shape, output_shape):
        """Add and remove the predictor to model MVar of wrong shape"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        a, b = input_shape
        assert b > 1
        with gp.Model() as gp_model:
            # Create a variable with wrong shape
            input_shape = a + 1, b + 1
            x = gp_model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            gp_model.update()
            gp_model.numvars

            gp_model.Params.OutputFlag = 0
            # All of these should fail
            with self.assertRaises(ValueError):
                # Both dimensions too big
                add_predictor_constr(gp_model, predictor, x, y)

    def add_remove_no_output(self, predictor, input_shape, output_shape):
        """Add and remove the predictor to model no output var"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as gp_model:
            x = gp_model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            gp_model.update()
            numvars = gp_model.numvars

            gp_model.Params.OutputFlag = 0
            pred_constr = add_predictor_constr(gp_model, predictor, x)

            with self.assertRaises(NoSolutionError):
                pred_constr.get_error()

            self.assertEqual(pred_constr.output.shape[0], output_shape[0])

            self.check_counts(gp_model, pred_constr, numvars)

            pred_constr.remove()
            gp_model.update()
            self.check_counts(gp_model, pred_constr, numvars)
            self.assertEqual(gp_model.NumConstrs, 0)
            self.assertEqual(gp_model.NumGenConstrs, 0)
            self.assertEqual(gp_model.NumQConstrs, 0)
            self.assertEqual(gp_model.NumVars, numvars)

    def add_remove_list_input(self, predictor, input_shape, output_shape):
        """Add and remove the predictor to model input and output var as lists."""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as gp_model:
            assert len(input_shape) == 2
            nexamples = input_shape[0]
            if len(output_shape) == 2:
                assert output_shape[0] == nexamples
                output_dim = output_shape[1]
            else:
                assert len(output_shape) == 1
                output_dim = 1
            pred_constrs = list()
            numvars = gp_model.numvars
            for _ in range(nexamples):
                varsbefore = gp_model.numvars
                x = gp_model.addVars(input_shape[1], lb=-gp.GRB.INFINITY)
                y = gp_model.addVars(output_dim, lb=-gp.GRB.INFINITY)
                gp_model.update()
                numvars += gp_model.numvars - varsbefore

                pred_constrs.append(add_predictor_constr(gp_model, predictor, x, y))

            for p2g in pred_constrs:
                with self.assertRaises(NoSolutionError):
                    p2g.get_error()

                p2g.remove()
            gp_model.update()
            self.assertEqual(gp_model.NumConstrs, 0)
            self.assertEqual(gp_model.NumGenConstrs, 0)
            self.assertEqual(gp_model.NumQConstrs, 0)
            self.assertEqual(gp_model.NumVars, numvars)

    def add_remove_list_of_lists_input(self, predictor, input_shape, output_shape):
        """Add and remove the predictor to model using list of lists"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as gp_model:
            x = gp_model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            gp_model.update()
            numvars = gp_model.numvars

            gp_model.Params.OutputFlag = 0
            pred_constr = add_predictor_constr(
                gp_model, predictor, x.tolist(), y.tolist()
            )

            self.check_counts(gp_model, pred_constr, numvars)

            with self.assertRaises(NoSolutionError):
                pred_constr.get_error()

            pred_constr.remove()
            gp_model.update()
            self.check_counts(gp_model, pred_constr, numvars)
            self.assertEqual(gp_model.NumConstrs, 0)
            self.assertEqual(gp_model.NumGenConstrs, 0)
            self.assertEqual(gp_model.NumQConstrs, 0)
            self.assertEqual(gp_model.NumVars, numvars)

    def add_remove_wrong_input_list_of_lists(
        self, predictor, input_shape, output_shape
    ):
        """Add and remove the predictor to model of wrong shape"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        a, b = input_shape
        assert b > 1
        with gp.Model() as gp_model:
            # Create a variable with wrong shape
            input_shape = a + 1, b + 1
            x = gp_model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            gp_model.update()
            gp_model.numvars

            gp_model.Params.OutputFlag = 0
            # All of these should fail
            with self.assertRaises(ValueError):
                # Both dimensions too big
                add_predictor_constr(gp_model, predictor, x.tolist(), y.tolist())

    def add_remove_non_rectangular_input(self, predictor, input_shape, output_shape):
        """Add and remove the predictor to model of wrong shape"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        a, b = input_shape
        assert b > 1
        with gp.Model() as gp_model:
            # Create a variable with wrong shape
            input_shape = a + 1, b + 1
            x = gp_model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            gp_model.update()
            gp_model.numvars

            gp_model.Params.OutputFlag = 0
            # All of these should fail
            with self.assertRaises(ValueError):
                # Both dimensions too big
                x = x.tolist()
                x[0] = x[0][1:]
                add_predictor_constr(gp_model, predictor, x, y.tolist())

    def test_diabetes_with_outputvar(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            onecase.pop("nonconvex")
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove(**onecase)

    def test_diabetes_without_outputvar(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        for regressor in cases:
            onecase = cases.get_case(regressor)
            onecase.pop("nonconvex")
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove_no_output(**onecase)

    def test_diabetes_varlist(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_list_input(**onecase)

    def test_diabetes_varlist_of_lists(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_list_of_lists_input(**onecase)

    def test_diabetes_wrong_input(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_wrong_input(**onecase)

    def test_diabetes_wrong_input_list(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_wrong_input_list_of_lists(**onecase)

    def test_diabetes_non_rectangular_input(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_non_rectangular_input(**onecase)

    def test_iris_with_outputvar(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisBinaryCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            onecase.pop("nonconvex")
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove(**onecase)

    def test_iris_without_outputvar(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisBinaryCases()
        for regressor in cases:
            onecase = cases.get_case(regressor)
            onecase.pop("nonconvex")
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove_no_output(**onecase)

    def test_iris_varlist(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisBinaryCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_list_input(**onecase)

    def test_iris_wrong_input(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisBinaryCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        onecase.pop("nonconvex")
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_wrong_input(**onecase)
