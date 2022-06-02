import unittest

import gurobipy as gp
from base_cases import DiabetesCases

from ml2gurobi.sklearn import pipe2gurobi


class TestFormulations(unittest.TestCase):

    def check_counts(self, m, reg2gurobi, numVars):
        self.assertEqual(m.NumVars, numVars + reg2gurobi.NumVars)
        self.assertEqual(m.NumSOS, reg2gurobi.NumSOS)
        self.assertEqual(m.NumConstrs, reg2gurobi.NumConstrs)
        self.assertEqual(m.NumQConstrs, reg2gurobi.NumQConstrs)
        self.assertEqual(m.NumGenConstrs, reg2gurobi.NumGenConstrs)

        self.assertEqual(reg2gurobi.NumVars, len(reg2gurobi.Vars))
        self.assertEqual(reg2gurobi.NumSOS, len(reg2gurobi.SOSs))
        self.assertEqual(reg2gurobi.NumConstrs, len(reg2gurobi.Constrs))
        self.assertEqual(reg2gurobi.NumQConstrs, len(reg2gurobi.QConstrs))
        self.assertEqual(reg2gurobi.NumGenConstrs, len(reg2gurobi.GenConstrs))

    def add_remove(self, predictor, translator, input_shape, output_shape):
        with gp.Model() as m:
            x = m.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = m.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            m.update()
            numVars = m.NumVars

            m.Params.OutputFlag = 0
            pred2grb = translator(m, predictor, x, y)

            self.check_counts(m, pred2grb, numVars)

            assert pred2grb.NumVars == len(pred2grb.Vars)

            pred2grb.remove()
            self.check_counts(m, pred2grb, numVars)
            assert m.NumConstrs == 0
            assert m.NumGenConstrs == 0
            assert m.NumQConstrs == 0

    def test_add_remove(self):
        cases = DiabetesCases()

        for regressor, translator in cases.to_test:
            for pipeline in [False, True]:
                case = cases.get_case(regressor, pipeline)
                if pipeline:
                    case['translator'] = pipe2gurobi
                else:
                    case['translator'] = translator
                with self.subTest(predictor=case['predictor'], pipeline=pipeline):
                    self.add_remove(**case)
