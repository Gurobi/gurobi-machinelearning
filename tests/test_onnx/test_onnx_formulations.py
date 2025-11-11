import os

import numpy as np
import onnx
from joblib import load
from onnx import helper, TensorProto

from ..fixed_formulation import FixedRegressionModel


def build_simple_mlp_onnx(
    n_in: int, n_hidden: int, n_out: int, seed: int = 0, use_gemm: bool = True
) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(n_in, n_hidden)).astype(np.float32)
    b1 = rng.normal(size=(n_hidden,)).astype(np.float32)
    W2 = rng.normal(size=(n_hidden, n_out)).astype(np.float32)
    b2 = rng.normal(size=(n_out,)).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n_out])

    if use_gemm:
        # Use Gemm operations
        init_W1 = helper.make_tensor(
            name="W1", data_type=TensorProto.FLOAT, dims=W1.T.shape, vals=W1.T.flatten()
        )
        init_b1 = helper.make_tensor(
            name="b1", data_type=TensorProto.FLOAT, dims=b1.shape, vals=b1
        )
        init_W2 = helper.make_tensor(
            name="W2", data_type=TensorProto.FLOAT, dims=W2.T.shape, vals=W2.T.flatten()
        )
        init_b2 = helper.make_tensor(
            name="b2", data_type=TensorProto.FLOAT, dims=b2.shape, vals=b2
        )

        gemm1 = helper.make_node(
            "Gemm", inputs=["X", "W1", "b1"], outputs=["H1"], name="gemm1", transB=1
        )
        relu1 = helper.make_node("Relu", inputs=["H1"], outputs=["A1"], name="relu1")
        gemm2 = helper.make_node(
            "Gemm", inputs=["A1", "W2", "b2"], outputs=["Y"], name="gemm2", transB=1
        )

        graph = helper.make_graph(
            nodes=[gemm1, relu1, gemm2],
            name="SimpleMLP",
            inputs=[X],
            outputs=[Y],
            initializer=[init_W1, init_b1, init_W2, init_b2],
        )
    else:
        # Use MatMul + Add operations (tf2onnx style)
        init_W1 = helper.make_tensor(
            name="W1", data_type=TensorProto.FLOAT, dims=W1.shape, vals=W1.flatten()
        )
        init_b1 = helper.make_tensor(
            name="b1", data_type=TensorProto.FLOAT, dims=b1.shape, vals=b1
        )
        init_W2 = helper.make_tensor(
            name="W2", data_type=TensorProto.FLOAT, dims=W2.shape, vals=W2.flatten()
        )
        init_b2 = helper.make_tensor(
            name="b2", data_type=TensorProto.FLOAT, dims=b2.shape, vals=b2
        )

        matmul1 = helper.make_node(
            "MatMul", inputs=["X", "W1"], outputs=["MM1"], name="matmul1"
        )
        add1 = helper.make_node(
            "Add", inputs=["MM1", "b1"], outputs=["H1"], name="add1"
        )
        relu1 = helper.make_node("Relu", inputs=["H1"], outputs=["A1"], name="relu1")
        matmul2 = helper.make_node(
            "MatMul", inputs=["A1", "W2"], outputs=["MM2"], name="matmul2"
        )
        add2 = helper.make_node("Add", inputs=["MM2", "b2"], outputs=["Y"], name="add2")

        graph = helper.make_graph(
            nodes=[matmul1, add1, relu1, matmul2, add2],
            name="SimpleMLP",
            inputs=[X],
            outputs=[Y],
            initializer=[init_W1, init_b1, init_W2, init_b2],
        )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_simple_cnn_onnx(seed: int = 42) -> onnx.ModelProto:
    """Build a simple CNN: Conv2d -> ReLU -> MaxPool -> Flatten -> Dense."""
    np.random.seed(seed)

    # Input: NCHW format, e.g., (1, 1, 8, 8) for single grayscale 8x8 image
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])

    # Conv layer: 1 input channel, 4 output channels, 3x3 kernel
    # After conv: (8-3+1) = 6x6x4
    conv_weight = np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.1
    conv_bias = np.random.randn(4).astype(np.float32) * 0.1

    init_conv_w = helper.make_tensor(
        name="conv_w",
        data_type=TensorProto.FLOAT,
        dims=conv_weight.shape,
        vals=conv_weight.flatten(),
    )
    init_conv_b = helper.make_tensor(
        name="conv_b", data_type=TensorProto.FLOAT, dims=conv_bias.shape, vals=conv_bias
    )

    # MaxPool 2x2: 6x6 -> 3x3
    # After flatten: 3*3*4 = 36
    # Dense layer: 36 -> 2
    dense_weight = np.random.randn(36, 2).astype(np.float32) * 0.1
    dense_bias = np.random.randn(2).astype(np.float32) * 0.1

    init_dense_w = helper.make_tensor(
        name="dense_w",
        data_type=TensorProto.FLOAT,
        dims=dense_weight.T.shape,
        vals=dense_weight.T.flatten(),
    )
    init_dense_b = helper.make_tensor(
        name="dense_b",
        data_type=TensorProto.FLOAT,
        dims=dense_bias.shape,
        vals=dense_bias,
    )

    # Build nodes
    conv = helper.make_node(
        "Conv",
        inputs=["X", "conv_w", "conv_b"],
        outputs=["conv_out"],
        name="conv",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    relu = helper.make_node(
        "Relu", inputs=["conv_out"], outputs=["relu_out"], name="relu"
    )
    maxpool = helper.make_node(
        "MaxPool",
        inputs=["relu_out"],
        outputs=["pool_out"],
        name="maxpool",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    flatten = helper.make_node(
        "Flatten", inputs=["pool_out"], outputs=["flat_out"], name="flatten", axis=1
    )
    dense = helper.make_node(
        "Gemm",
        inputs=["flat_out", "dense_w", "dense_b"],
        outputs=["Y"],
        name="dense",
        transB=1,
    )

    graph = helper.make_graph(
        nodes=[conv, relu, maxpool, flatten, dense],
        name="SimpleCNN",
        inputs=[X],
        outputs=[Y],
        initializer=[init_conv_w, init_conv_b, init_dense_w, init_dense_b],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


class TestONNXModel(FixedRegressionModel):
    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_onnx_mlp(self):
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))
        n_in = X.shape[1]
        model = build_simple_mlp_onnx(n_in=n_in, n_hidden=16, n_out=1, seed=123)
        onecase = {"predictor": model, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")

    def test_diabetes_onnx_mlp_matmul(self):
        """Test ONNX models using MatMul+Add pattern (tf2onnx style)."""
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))
        n_in = X.shape[1]
        model = build_simple_mlp_onnx(
            n_in=n_in, n_hidden=16, n_out=1, seed=123, use_gemm=False
        )
        onecase = {"predictor": model, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")

    def test_onnx_cnn(self):
        """Test ONNX CNN model with Conv2d, MaxPool, and Flatten."""
        import gurobipy as gp
        import onnxruntime as ort

        # Create synthetic image data (batch of 8x8 images in NCHW for ONNX)
        np.random.seed(42)
        X_nchw = np.random.randn(5, 1, 8, 8).astype(np.float32)

        model = build_simple_cnn_onnx(seed=123)

        # Test with ONNX Runtime to get ground truth
        sess = ort.InferenceSession(model.SerializeToString())

        # Test a few samples
        for idx in [0, 2, 4]:
            sample_nchw = X_nchw[idx : idx + 1]
            sess.run(None, {"X": sample_nchw})[0]

            # Convert to NHWC for our system
            sample_nhwc = np.transpose(sample_nchw, (0, 2, 3, 1))

            # Create Gurobi model
            gpm = gp.Model()
            gpm.setParam("OutputFlag", 0)
            x = gpm.addMVar(
                sample_nhwc.shape, lb=sample_nhwc - 1e-4, ub=sample_nhwc + 1e-4
            )

            from gurobi_ml.onnx import add_onnx_constr

            pred_constr = add_onnx_constr(gpm, model, x)

            gpm.optimize()

            self.assertEqual(gpm.status, gp.GRB.OPTIMAL)

            # Check error
            error = pred_constr.get_error()
            max_error = np.max(np.abs(error))
            self.assertLess(
                max_error,
                1e-5,
                f"Sample {idx}: max error {max_error} exceeds tolerance",
            )

    def test_onnx_cnn_with_padding(self):
        """Test ONNX CNN model with Conv2d using padding."""
        import gurobipy as gp
        import onnxruntime as ort

        # Create a simple Conv model with padding
        np.random.seed(42)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 5, 5])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2, 5, 5])

        # Conv with padding=1 to maintain spatial dimensions
        conv_weight = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.1
        conv_bias = np.random.randn(2).astype(np.float32) * 0.1

        init_conv_w = helper.make_tensor(
            name="conv_w",
            data_type=TensorProto.FLOAT,
            dims=conv_weight.shape,
            vals=conv_weight.flatten(),
        )
        init_conv_b = helper.make_tensor(
            name="conv_b",
            data_type=TensorProto.FLOAT,
            dims=conv_bias.shape,
            vals=conv_bias,
        )

        conv = helper.make_node(
            "Conv",
            inputs=["X", "conv_w", "conv_b"],
            outputs=["Y"],
            name="conv",
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],  # Symmetric padding
        )

        graph = helper.make_graph(
            nodes=[conv],
            name="ConvWithPadding",
            inputs=[X],
            outputs=[Y],
            initializer=[init_conv_w, init_conv_b],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)

        # Test with ONNX Runtime
        sess = ort.InferenceSession(model.SerializeToString())
        test_input_nchw = np.random.randn(1, 1, 5, 5).astype(np.float32)
        sess.run(None, {"X": test_input_nchw})[0]

        # Test with Gurobi
        gpm = gp.Model()
        gpm.setParam("OutputFlag", 0)

        test_input_nhwc = np.transpose(test_input_nchw, (0, 2, 3, 1))
        x = gpm.addMVar(
            test_input_nhwc.shape, lb=test_input_nhwc - 1e-4, ub=test_input_nhwc + 1e-4
        )

        from gurobi_ml.onnx import add_onnx_constr

        pred_constr = add_onnx_constr(gpm, model, x)

        gpm.optimize()

        self.assertEqual(gpm.status, gp.GRB.OPTIMAL)

        # Check error
        error = pred_constr.get_error()
        max_error = np.max(np.abs(error))
        self.assertLess(max_error, 1e-5, f"Max error {max_error} exceeds tolerance")
