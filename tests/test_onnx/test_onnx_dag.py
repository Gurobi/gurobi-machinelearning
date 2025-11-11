import os
import unittest

import gurobipy as gp
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from gurobi_ml.onnx import add_onnx_dag_constr


class TestONNXDAG(unittest.TestCase):
    """Tests for DAG-based ONNX implementation."""

    def test_sequential_model(self):
        """Test that sequential models work with DAG implementation."""
        n_in, n_hidden, n_out = 4, 8, 2
        W1 = np.random.randn(n_in, n_hidden).astype(np.float32)
        b1 = np.random.randn(n_hidden).astype(np.float32)
        W2 = np.random.randn(n_hidden, n_out).astype(np.float32)
        b2 = np.random.randn(n_out).astype(np.float32)

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

        gemm1 = helper.make_node("Gemm", ["X", "W1", "b1"], ["H1"], transB=1)
        relu1 = helper.make_node("Relu", ["H1"], ["A1"])
        gemm2 = helper.make_node("Gemm", ["A1", "W2", "b2"], ["Y"], transB=1)

        graph = helper.make_graph(
            [gemm1, relu1, gemm2],
            "SequentialMLP",
            [X],
            [Y],
            [init_W1, init_b1, init_W2, init_b2],
        )
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)]
        )
        onnx_model.ir_version = 9

        # Test with ONNX Runtime
        test_input = np.random.randn(1, n_in).astype(np.float32)
        ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_output = ort_sess.run(None, {"X": test_input})[0]

        # Test with Gurobi
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        x = m.addMVar((n_in,), lb=-10, ub=10, name="x")
        for i in range(n_in):
            x[i].lb = x[i].ub = test_input[0, i]

        pred = add_onnx_dag_constr(m, onnx_model, x, None)
        m.optimize()

        self.assertEqual(m.status, gp.GRB.OPTIMAL)
        gurobi_output = pred.output.X.reshape(-1)
        error = np.abs(onnx_output[0] - gurobi_output).max()
        self.assertLess(error, 1e-5)

    def test_skip_connection(self):
        """Test model with skip connection (input used by multiple layers)."""
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
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)]
        )
        onnx_model.ir_version = 9

        # Test with ONNX Runtime
        test_input = np.random.randn(1, n_in).astype(np.float32)
        ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_output = ort_sess.run(None, {"X": test_input})[0]

        # Test with Gurobi DAG
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        x = m.addMVar((n_in,), lb=-10, ub=10, name="x")
        for i in range(n_in):
            x[i].lb = x[i].ub = test_input[0, i]

        pred = add_onnx_dag_constr(m, onnx_model, x, None)
        m.optimize()

        self.assertEqual(m.status, gp.GRB.OPTIMAL)
        gurobi_output = pred.output.X.reshape(-1)
        error = np.abs(onnx_output[0] - gurobi_output).max()
        self.assertLess(error, 1e-5)

    def test_residual_connection(self):
        """Test model with residual connection (intermediate value reused)."""
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
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)]
        )
        onnx_model.ir_version = 9

        # Test with ONNX Runtime
        test_input = np.random.randn(1, n_in).astype(np.float32)
        ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_output = ort_sess.run(None, {"X": test_input})[0]

        # Test with Gurobi DAG
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        x = m.addMVar((n_in,), lb=-10, ub=10, name="x")
        for i in range(n_in):
            x[i].lb = x[i].ub = test_input[0, i]

        pred = add_onnx_dag_constr(m, onnx_model, x, None)
        m.optimize()

        self.assertEqual(m.status, gp.GRB.OPTIMAL)
        gurobi_output = pred.output.X.reshape(-1)
        error = np.abs(onnx_output[0] - gurobi_output).max()
        self.assertLess(error, 1e-5)

    def test_real_world_models(self):
        """Test with real CerSyVe models that have skip/residual connections."""
        basedir = os.path.dirname(__file__)
        workspace_dir = os.path.join(basedir, "..", "..")

        models_to_test = [
            ("pendulum_pretrain_con.onnx", 2, 2),
            ("double_integrator_pretrain_con.onnx", 2, 2),
        ]

        for model_name, input_dim, output_dim in models_to_test:
            model_path = os.path.join(workspace_dir, model_name)
            if not os.path.exists(model_path):
                continue  # Skip if model not found

            with self.subTest(model=model_name):
                # Load model
                onnx_model = onnx.load(model_path)

                # Test with random input
                np.random.seed(42)
                test_input = np.random.randn(1, input_dim).astype(np.float32)

                # ONNX Runtime
                ort_sess = ort.InferenceSession(model_path)
                input_name = ort_sess.get_inputs()[0].name
                onnx_output = ort_sess.run(None, {input_name: test_input})[0]

                # Gurobi ML DAG
                m = gp.Model()
                m.setParam("OutputFlag", 0)
                x = m.addMVar((input_dim,), name="x")
                for i in range(input_dim):
                    x[i].lb = x[i].ub = test_input[0, i]

                pred = add_onnx_dag_constr(m, onnx_model, x, None)
                m.optimize()

                self.assertEqual(m.status, gp.GRB.OPTIMAL)
                gurobi_output = pred.output.X.reshape(-1)
                error = np.abs(onnx_output[0] - gurobi_output).max()
                self.assertLess(error, 1e-4)


if __name__ == "__main__":
    unittest.main()
