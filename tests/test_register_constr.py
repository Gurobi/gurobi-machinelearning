""" Test register_predictor_constr function"""

import unittest

import gurobipy as gp
from sklearn.svm import LinearSVC

from gurobi_ml.add_predictor import add_predictor_constr, register_predictor_constr
from gurobi_ml.base import AbstractPredictorConstr


class DummyPredictorError(Exception):
    def __init__(self):
        super().__init__("This is the exception I want to see")


class DummyPredictor(AbstractPredictorConstr):
    def __init__(self, model, predictor, input_vars, output_vars, **kwargs):
        super().__init__(model, input_vars, output_vars, **kwargs)

    def mip_model(self, *args, **kwargs):
        raise DummyPredictorError()


class TestDummyConstr(unittest.TestCase):
    def test_register_predictor(self):
        register_predictor_constr(LinearSVC, DummyPredictor)

        with gp.Model() as m:
            inp = m.addVar()
            out = m.addVar()

            sbc = LinearSVC()

            with self.assertRaises(DummyPredictorError):
                add_predictor_constr(m, sbc, inp, out)
