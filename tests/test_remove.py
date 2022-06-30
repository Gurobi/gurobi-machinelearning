import unittest

import gurobipy as gp
from base_cases import DiabetesCases

from ml2gurobi.sklearn import PipelinePredictor


class TestFormulations(unittest.TestCase):

    def check_counts(self, m, reg2gurobi, numVars):
        self.assertEqual(m.NumVars, numVars + reg2gurobi.get_num('Vars'))
        self.assertEqual(m.NumSOS, reg2gurobi.get_num('SOS'))
        self.assertEqual(m.NumConstrs, reg2gurobi.get_num('Constrs'))
        self.assertEqual(m.NumQConstrs, reg2gurobi.get_num('QConstrs'))
        self.assertEqual(m.NumGenConstrs, reg2gurobi.get_num('GenConstrs'))

        self.assertEqual(reg2gurobi.get_num('Vars'), len(reg2gurobi.get_list('Vars')))
        self.assertEqual(reg2gurobi.get_num('SOS'), len(reg2gurobi.get_list('SOS')))
        self.assertEqual(reg2gurobi.get_num('Constrs'), len(reg2gurobi.get_list('Constrs')))
        self.assertEqual(reg2gurobi.get_num('QConstrs'), len(reg2gurobi.get_list('QConstrs')))
        self.assertEqual(reg2gurobi.get_num('GenConstrs'), len(reg2gurobi.get_list('GenConstrs')))

    def add_remove(self, predictor, translator, input_shape, output_shape):
        with gp.Model() as m:
            x = m.addMVar(input_shape, lb=-gp.GRB.INFINITY)
            y = m.addMVar(output_shape, lb=-gp.GRB.INFINITY)
            m.update()
            numVars = m.NumVars

            m.Params.OutputFlag = 0
            pred2grb = translator(m, predictor, x, y)

            self.check_counts(m, pred2grb, numVars)

            assert pred2grb.get_num('Vars') == len(pred2grb.get_list('Vars'))

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
                    case['translator'] = PipelinePredictor
                else:
                    case['translator'] = translator
                with self.subTest(predictor=case['predictor'], pipeline=pipeline):
                    self.add_remove(**case)
