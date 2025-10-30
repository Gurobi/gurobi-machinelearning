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
