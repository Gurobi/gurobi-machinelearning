import unittest

import gurobipy as gp
from base_cases import DiabetesCases

from ml2gurobi.sklearn import PipelinePredictor


class TestFormulations(unittest.TestCase):

    def check_counts(self, m, reg2gurobi, numVars):
        self.assertEqual(m.NumVars, numVars + len(reg2gurobi.getVars()))
        self.assertEqual(m.NumSOS, len(reg2gurobi.getSOSs()))
        self.assertEqual(m.NumConstrs, len(reg2gurobi.getConstrs()))
        self.assertEqual(m.NumQConstrs, len(reg2gurobi.getQConstrs()))
        self.assertEqual(m.NumGenConstrs, len(reg2gurobi.getGenConstrs()))


    def add_remove(self, predictor, translator, input_shape, output_shape):
        with gp.Model() as m:
            x = m.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = m.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            m.update()
            numVars = m.NumVars

            m.Params.OutputFlag = 0
            pred2grb = translator(m, predictor, x, y)

            self.check_counts(m, pred2grb, numVars)

            pred2grb.remove()
            m.update()
            self.check_counts(m, pred2grb, numVars)
            assert m.NumConstrs == 0
            assert m.NumGenConstrs == 0
            assert m.NumQConstrs == 0
            assert m.NumVars == numVars

    def test_add_remove(self):
        cases = DiabetesCases()

        for regressor, translator in cases.to_test:
            for pipeline in [False, True]:
                case = cases.get_case(regressor, pipeline)
                if pipeline:
                    case['translator'] = PipelinePredictor
                else:
                    case['translator'] = translator
                with self.subTest(predictor=case['predictor'], pipeline=pipeline):
                    self.add_remove(**case)
