import unittest

import gurobipy as gp
import numpy as np
import onnx
from onnx import helper, TensorProto

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import ModelConfigurationError


class TestUnsupportedONNX(unittest.TestCase):
    def test_unsupported_op(self):
        # Build a simple graph with an unsupported op (Sigmoid)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])
        node = helper.make_node("Sigmoid", inputs=["X"], outputs=["Y"], name="sigmoid")
        graph = helper.make_graph(
            nodes=[node], name="BadGraph", inputs=[X], outputs=[Y]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)

        example = np.zeros((1, 4), dtype=float)
        m = gp.Model()
        x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")
        with self.assertRaises(ModelConfigurationError):
            add_predictor_constr(m, model, x)

    def test_skip_connection_rejected(self):
        # Build a model with skip connection: input used by multiple nodes
        n_in, n_hidden, n_out = 4, 8, 2

        W1 = np.random.randn(n_in, n_hidden).astype(np.float32)
        b1 = np.random.randn(n_hidden).astype(np.float32)
        W2 = np.random.randn(n_hidden, n_out).astype(np.float32)
        b2 = np.random.randn(n_out).astype(np.float32)
        W_skip = np.random.randn(n_in, n_out).astype(np.float32)
        b_skip = np.random.randn(n_out).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n_out])

        init_W1 = helper.make_tensor(
            "W1", TensorProto.FLOAT, W1.T.shape, W1.T.flatten()
        )
        init_b1 = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1)
        init_W2 = helper.make_tensor(
            "W2", TensorProto.FLOAT, W2.T.shape, W2.T.flatten()
        )
        init_b2 = helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2)
        init_W_skip = helper.make_tensor(
            "W_skip", TensorProto.FLOAT, W_skip.T.shape, W_skip.T.flatten()
        )
        init_b_skip = helper.make_tensor(
            "b_skip", TensorProto.FLOAT, b_skip.shape, b_skip
        )

        # Main path
        gemm1 = helper.make_node("Gemm", ["X", "W1", "b1"], ["H1"], transB=1)
        relu1 = helper.make_node("Relu", ["H1"], ["A1"])
        gemm2 = helper.make_node("Gemm", ["A1", "W2", "b2"], ["branch1"], transB=1)

        # Skip connection path - uses X again!
        gemm_skip = helper.make_node(
            "Gemm", ["X", "W_skip", "b_skip"], ["branch2"], transB=1
        )

        # Combine branches (residual add)
        add = helper.make_node("Add", ["branch1", "branch2"], ["Y"])

        graph = helper.make_graph(
            [gemm1, relu1, gemm2, gemm_skip, add],
            "SkipConnectionMLP",
            [X],
            [Y],
            [init_W1, init_b1, init_W2, init_b2, init_W_skip, init_b_skip],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)

        m = gp.Model()
        x = m.addMVar((n_in,), lb=-1.0, ub=1.0, name="x")
        with self.assertRaises(ModelConfigurationError) as cm:
            add_predictor_constr(m, model, x)

        # Verify the error message mentions skip connections
        self.assertIn("skip connection", str(cm.exception).lower())

    def test_residual_connection_rejected(self):
        # Build a model with residual connection: intermediate value used by multiple nodes
        n_in, n_hidden, n_out = 4, 8, 2

        W1 = np.random.randn(n_in, n_hidden).astype(np.float32)
        b1 = np.random.randn(n_hidden).astype(np.float32)
        W2a = np.random.randn(n_hidden, n_out).astype(np.float32)
        b2a = np.random.randn(n_out).astype(np.float32)
        W2b = np.random.randn(n_hidden, n_out).astype(np.float32)
        b2b = np.random.randn(n_out).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n_out])

        init_W1 = helper.make_tensor(
            "W1", TensorProto.FLOAT, W1.T.shape, W1.T.flatten()
        )
        init_b1 = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1)
        init_W2a = helper.make_tensor(
            "W2a", TensorProto.FLOAT, W2a.T.shape, W2a.T.flatten()
        )
        init_b2a = helper.make_tensor("b2a", TensorProto.FLOAT, b2a.shape, b2a)
        init_W2b = helper.make_tensor(
            "W2b", TensorProto.FLOAT, W2b.T.shape, W2b.T.flatten()
        )
        init_b2b = helper.make_tensor("b2b", TensorProto.FLOAT, b2b.shape, b2b)

        # Shared layer
        gemm1 = helper.make_node("Gemm", ["X", "W1", "b1"], ["H1"], transB=1)
        relu1 = helper.make_node("Relu", ["H1"], ["A1"])

        # Branch 1 - uses A1
        gemm2a = helper.make_node("Gemm", ["A1", "W2a", "b2a"], ["branch1"], transB=1)

        # Branch 2 - also uses A1!
        gemm2b = helper.make_node("Gemm", ["A1", "W2b", "b2b"], ["branch2"], transB=1)

        # Combine branches
        add = helper.make_node("Add", ["branch1", "branch2"], ["Y"])

        graph = helper.make_graph(
            [gemm1, relu1, gemm2a, gemm2b, add],
            "ResidualMLP",
            [X],
            [Y],
            [init_W1, init_b1, init_W2a, init_b2a, init_W2b, init_b2b],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)

        m = gp.Model()
        x = m.addMVar((n_in,), lb=-1.0, ub=1.0, name="x")
        with self.assertRaises(ModelConfigurationError) as cm:
            add_predictor_constr(m, model, x)

        # Verify the error message mentions the architecture issue
        self.assertIn("non-sequential", str(cm.exception).lower())
