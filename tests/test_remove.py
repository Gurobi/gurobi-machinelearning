""" Test for adding and removing predictor objects"""
import unittest

import gurobipy as gp
from base_cases import DiabetesCases

from ml2gurobi.sklearn import PipelineConstr


class TestAddRemove(unittest.TestCase):
    """Test Adding and Removing submodels and check counts."""

    def check_counts(self, model, reg2gurobi, numvars):
        """Assert counts are ok"""
        self.assertEqual(model.numvars, numvars + len(reg2gurobi.getVars()))
        self.assertEqual(model.numsos, len(reg2gurobi.getSOSs()))
        self.assertEqual(model.numconstrs, len(reg2gurobi.getConstrs()))
        self.assertEqual(model.numqconstrs, len(reg2gurobi.getQConstrs()))
        self.assertEqual(model.numgenconstrs, len(reg2gurobi.getGenConstrs()))

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
            assert model.NumConstrs == 0
            assert model.NumGenConstrs == 0
            assert model.NumQConstrs == 0
            assert model.NumVars == numvars

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
                with self.subTest(predictor=case["predictor"], pipeline=pipeline):
                    self.add_remove(**case)
