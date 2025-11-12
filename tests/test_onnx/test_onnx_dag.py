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

    def test_cnn_with_residual(self):
        """Test CNN with Conv, Flatten, and skip connection."""
        batch_size = 1
        in_channels = 1
        height, width = 4, 4
        out_channels = 2
        kernel_size = 2

        # Conv layer
        conv_weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32)
        conv_bias = np.random.randn(out_channels).astype(np.float32)

        # Calculate sizes
        conv_out_h = height - kernel_size + 1
        conv_out_w = width - kernel_size + 1
        flatten_size = conv_out_h * conv_out_w * out_channels
        input_flatten_size = height * width * in_channels

        # Dense layers
        dense1_out = 4
        dense1_weights = np.random.randn(flatten_size, dense1_out).astype(np.float32)
        dense1_bias = np.random.randn(dense1_out).astype(np.float32)

        # Skip connection dense
        skip_weights = np.random.randn(input_flatten_size, dense1_out).astype(
            np.float32
        )
        skip_bias = np.random.randn(dense1_out).astype(np.float32)

        # Final dense
        final_out = 2
        final_weights = np.random.randn(dense1_out, final_out).astype(np.float32)
        final_bias = np.random.randn(final_out).astype(np.float32)

        # Create ONNX model
        X = helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, [None, in_channels, height, width]
        )
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, final_out])

        init_conv_w = helper.make_tensor(
            "conv_w", TensorProto.FLOAT, conv_weights.shape, conv_weights.flatten()
        )
        init_conv_b = helper.make_tensor(
            "conv_b", TensorProto.FLOAT, conv_bias.shape, conv_bias
        )
        init_dense1_w = helper.make_tensor(
            "dense1_w",
            TensorProto.FLOAT,
            dense1_weights.T.shape,
            dense1_weights.T.flatten(),
        )
        init_dense1_b = helper.make_tensor(
            "dense1_b", TensorProto.FLOAT, dense1_bias.shape, dense1_bias
        )
        init_skip_w = helper.make_tensor(
            "skip_w", TensorProto.FLOAT, skip_weights.T.shape, skip_weights.T.flatten()
        )
        init_skip_b = helper.make_tensor(
            "skip_b", TensorProto.FLOAT, skip_bias.shape, skip_bias
        )
        init_final_w = helper.make_tensor(
            "final_w",
            TensorProto.FLOAT,
            final_weights.T.shape,
            final_weights.T.flatten(),
        )
        init_final_b = helper.make_tensor(
            "final_b", TensorProto.FLOAT, final_bias.shape, final_bias
        )

        # Main path: Conv -> Flatten -> Dense
        conv = helper.make_node("Conv", ["X", "conv_w", "conv_b"], ["conv_out"])
        flatten1 = helper.make_node("Flatten", ["conv_out"], ["flatten1_out"], axis=1)
        dense1 = helper.make_node(
            "Gemm", ["flatten1_out", "dense1_w", "dense1_b"], ["branch1"], transB=1
        )

        # Skip path: Flatten input -> Dense
        flatten2 = helper.make_node("Flatten", ["X"], ["flatten2_out"], axis=1)
        skip_dense = helper.make_node(
            "Gemm", ["flatten2_out", "skip_w", "skip_b"], ["branch2"], transB=1
        )

        # Combine branches
        add = helper.make_node("Add", ["branch1", "branch2"], ["added"])
        relu = helper.make_node("Relu", ["added"], ["relu_out"])
        final = helper.make_node(
            "Gemm", ["relu_out", "final_w", "final_b"], ["Y"], transB=1
        )

        graph = helper.make_graph(
            [conv, flatten1, dense1, flatten2, skip_dense, add, relu, final],
            "CNNWithSkip",
            [X],
            [Y],
            [
                init_conv_w,
                init_conv_b,
                init_dense1_w,
                init_dense1_b,
                init_skip_w,
                init_skip_b,
                init_final_w,
                init_final_b,
            ],
        )
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)]
        )
        onnx_model.ir_version = 9

        # Test with random input (NCHW for ONNX)
        np.random.seed(42)
        test_input_nchw = np.random.randn(
            batch_size, in_channels, height, width
        ).astype(np.float32)

        # ONNX Runtime
        ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_output = ort_sess.run(None, {"X": test_input_nchw})[0]

        # Gurobi ML DAG (expects NHWC)
        test_input_nhwc = np.transpose(test_input_nchw, (0, 2, 3, 1))

        m = gp.Model()
        m.setParam("OutputFlag", 0)
        x = m.addMVar(test_input_nhwc.shape, name="x")
        for idx in np.ndindex(test_input_nhwc.shape):
            x[idx].lb = x[idx].ub = test_input_nhwc[idx]

        pred = add_onnx_dag_constr(m, onnx_model, x, None)
        m.optimize()

        self.assertEqual(m.status, gp.GRB.OPTIMAL)
        gurobi_output = pred.output.X
        if gurobi_output.ndim == 1:
            gurobi_output = gurobi_output.reshape(1, -1)

        error = np.abs(onnx_output - gurobi_output).max()
        self.assertLess(error, 1e-4)


if __name__ == "__main__":
    unittest.main()
