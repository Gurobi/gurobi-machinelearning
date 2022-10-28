""" Test for adding and removing predictor objects"""
import random
import unittest

import gurobipy as gp
from base_cases import DiabetesCases, IrisCases

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoSolution, ParameterError


class TestAddRemove(unittest.TestCase):
    """Test Adding and Removing submodels and check counts."""

    def check_counts(self, model, reg2gurobi, numvars):
        """Assert counts are ok"""
        self.assertEqual(model.NumVars, numvars + len(reg2gurobi.vars))
        self.assertEqual(model.NumSOS, len(reg2gurobi.sos))
        self.assertEqual(model.NumConstrs, len(reg2gurobi.constrs))
        self.assertEqual(model.NumQConstrs, len(reg2gurobi.qconstrs))
        self.assertEqual(model.NumGenConstrs, len(reg2gurobi.genconstrs))

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

    def add_remove(self, predictor, input_shape, output_shape, nonconvex):
        """Add and remove the predictor to model"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as model:
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            pred2grb = add_predictor_constr(model, predictor, x, y)

            self.check_counts(model, pred2grb, numvars)

            with self.assertRaises(NoSolution):
                pred2grb.get_error()

            pred2grb.remove()
            model.update()
            self.check_counts(model, pred2grb, numvars)
            self.assertEqual(model.NumConstrs, 0)
            self.assertEqual(model.NumGenConstrs, 0)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertEqual(model.NumVars, numvars)

    def add_remove_wrong_input(self, predictor, input_shape, output_shape, nonconvex):
        """Add and remove the predictor to model"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        a, b = input_shape
        assert b > 1
        with gp.Model() as model:
            # Create a variable with wrong shape
            input_shape = a + 1, b + 1
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            with self.assertRaises(ParameterError):
                # All of these should fail

                # Both dimensions too big
                pred2grb = add_predictor_constr(model, predictor, x, y)

                # Second dimension (features) too big
                pred2grb = add_predictor_constr(model, predictor, x.reshape(a, b + 1), y)

                # Second dimension (features too small
                pred2grb = add_predictor_constr(model, predictor, x.reshape(a, b - 1), y)

                # Empty input variable
                pred2grb = add_predictor_constr(model, predictor, None, y)

                # Mismatch in input output dimensions
                pred2grb = add_predictor_constr(model, predictor, x.reshape(a - 1, b), y)

    def add_remove_no_output(self, predictor, input_shape, output_shape, nonconvex):
        """Add and remove the predictor to model"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as model:
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            pred2grb = add_predictor_constr(model, predictor, x)

            with self.assertRaises(NoSolution):
                pred2grb.get_error()

            self.assertEqual(pred2grb.output.shape[0], output_shape[0])

            self.check_counts(model, pred2grb, numvars)

            pred2grb.remove()
            model.update()
            self.check_counts(model, pred2grb, numvars)
            self.assertEqual(model.NumConstrs, 0)
            self.assertEqual(model.NumGenConstrs, 0)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertEqual(model.NumVars, numvars)

    def add_remove_list_input(self, predictor, input_shape, output_shape, nonconvex):
        """Add and remove the predictor to model"""
        input_shape, output_shape = self.truncate_shapes(input_shape, output_shape)
        with gp.Model() as model:
            assert len(input_shape) == 2
            nexamples = input_shape[0]
            if len(output_shape) == 2:
                assert output_shape[0] == nexamples
                output_dim = output_shape[1]
            else:
                assert len(output_shape) == 1
                output_dim = 1
            pred2grb = list()
            numvars = model.numvars
            for k in range(nexamples):
                varsbefore = model.numvars
                x = model.addVars(input_shape[1], lb=-gp.GRB.INFINITY)
                y = model.addVars(output_dim, lb=-gp.GRB.INFINITY)
                model.update()
                numvars += model.numvars - varsbefore

                pred2grb.append(add_predictor_constr(model, predictor, x, y))

            for p2g in pred2grb:
                with self.assertRaises(NoSolution):
                    p2g.get_error()

                p2g.remove()
            model.update()
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

    def test_diabetes_varlist(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_list_input(**onecase)

    def test_diabetes_wrong_input(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_wrong_input(**onecase)

    def test_iris_with_outputvar(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove(**onecase)

    def test_iris_without_outputvar(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisCases()
        for regressor in cases:
            onecase = cases.get_case(regressor)
            with self.subTest(predictor=onecase["predictor"]):
                self.add_remove_no_output(**onecase)

    def test_iris_varlist(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_list_input(**onecase)

    def test_iris_wrong_input(self):
        """Test adding and removing a predictor for iris

        Checks that variables/constraints/... counts match.
        """
        cases = IrisCases()
        regressor = random.choice(list(cases))

        onecase = cases.get_case(regressor)
        with self.subTest(predictor=onecase["predictor"]):
            self.add_remove_wrong_input(**onecase)
