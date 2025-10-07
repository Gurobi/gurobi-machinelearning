import unittest

import gurobipy as gp
import numpy as np
import onnx
from onnx import helper, TensorProto

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoModel


class TestUnsupportedONNX(unittest.TestCase):
    def test_unsupported_op(self):
        # Build a simple graph with an unsupported op (Sigmoid)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])
        node = helper.make_node("Sigmoid", inputs=["X"], outputs=["Y"], name="sigmoid")
        graph = helper.make_graph(
            nodes=[node], name="BadGraph", inputs=[X], outputs=[Y]
        )
        model = helper.make_model(graph)
        onnx.checker.check_model(model)

        example = np.zeros((1, 4), dtype=float)
        m = gp.Model()
        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")
        with self.assertRaises(NoModel):
            add_predictor_constr(m, model, x)
