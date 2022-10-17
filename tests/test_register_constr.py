# Copyright © 2022 Gurobi Optimization, LLC
""" Test register_predictor_constr function"""

import unittest

import gurobipy as gp
from sklearn.svm import LinearSVC, LinearSVR

from gurobi_ml.add_predictor import add_predictor_constr, register_predictor_constr
from gurobi_ml.exceptions import NotRegistered
from gurobi_ml.modeling import AbstractPredictorConstr


class DummyPredictorError(Exception):
    def __init__(self):
        super().__init__("This is the exception I want to see")


class DummyPredictor(AbstractPredictorConstr):
    def __init__(self, model, predictor, input_vars, output_vars, **kwargs):
        super().__init__(model, input_vars, output_vars, **kwargs)

    def _mip_model(self, *args, **kwargs):
        raise DummyPredictorError()


class TestDummyConstr(unittest.TestCase):
    def test_register_predictor(self):
        register_predictor_constr(LinearSVC, DummyPredictor)

        with gp.Model() as m:
            inp = m.addVar()
            out = m.addVar()

            svc = LinearSVC()

            with self.assertRaises(DummyPredictorError):
                add_predictor_constr(m, svc, inp, out)

    def test_unregeistered_predictor_constr(self):
        with gp.Model() as m:
            inp = m.addVar()
            out = m.addVar()

            svr = LinearSVR()

            with self.assertRaises(NotRegistered):
                add_predictor_constr(m, svr, inp, out)
