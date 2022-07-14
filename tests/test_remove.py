""" Test for adding and removing predictor objects"""
import unittest

import gurobipy as gp
from base_cases import DiabetesCases

from ml2gurobi import add_predictor_constr
from ml2gurobi.sklearn import PipelineConstr


class TestAddRemove(unittest.TestCase):
    """Test Adding and Removing submodels and check counts."""

    def check_counts(self, model, reg2gurobi, numvars):
        """Assert counts are ok"""
        self.assertEqual(m.NumVars, numVars + len(reg2gurobi.vars))
        self.assertEqual(m.NumSOS, len(reg2gurobi.sos))
        self.assertEqual(m.NumConstrs, len(reg2gurobi.constrs))
        self.assertEqual(m.NumQConstrs, len(reg2gurobi.qconstrs))
        self.assertEqual(m.NumGenConstrs, len(reg2gurobi.genconstrs))

    def add_remove(self, predictor, translator, input_shape, output_shape):
        """Add and remove the predictor to model"""
        with gp.Model() as model:
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = model.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            pred2grb = translator(model, predictor, x, y)

            self.check_counts(model, pred2grb, numvars)

            pred2grb.remove()
            model.update()
            self.check_counts(model, pred2grb, numvars)
            self.assertEqual(model.NumConstrs, 0)
            self.assertEqual(model.NumGenConstrs, 0)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertEqual(model.NumVars, numvars)

    def add_remove_no_output(self, translator, predictor, input_shape, output_shape):
        """Add and remove the predictor to model"""
        with gp.Model() as model:
            x = model.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            model.update()
            numvars = model.numvars

            model.Params.OutputFlag = 0
            pred2grb = add_predictor_constr(model, predictor, x)

            self.assertIsInstance(pred2grb, predictor)
            self.assertEqual(pred2grb.output.shape, output_shape)

            self.check_counts(model, pred2grb, numvars)

            pred2grb.remove()
            model.update()
            self.check_counts(model, pred2grb, numvars)
            self.assertEqual(model.NumConstrs, 0)
            self.assertEqual(model.NumGenConstrs, 0)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertEqual(model.NumVars, numvars)

    def test_diabetes(self):
        """Test adding and removing a predictor for diabetes

        Checks that variables/constraints/... counts match.
        """
        cases = DiabetesCases()

        for regressor, translator in cases.to_test:
            for pipeline in [False, True]:
                case = cases.get_case(regressor, pipeline)
                if pipeline:
                    case["translator"] = PipelineConstr
                else:
                    case["translator"] = translator
                with self.subTest(predictor=case["predictor"], outputvar=True, pipeline=pipeline):
                    self.add_remove(**case)
                with self.subTest(predictor=case["predictor"], outputvar=False, pipeline=pipeline):
                    self.add_remove(**case)
