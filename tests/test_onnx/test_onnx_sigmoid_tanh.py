"""Tests for ONNX Sigmoid and Tanh activation support."""

import numpy as np
import pytest
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

import gurobipy as gp
from gurobi_ml import add_predictor_constr
from gurobi_ml._grb_version import HAS_NLFUNC, HAS_TANH


def _build_onnx_mlp(activation_op: str, n_in=3, n_hidden=4, n_out=1, seed=42):
    """Build a minimal ONNX MLP with the given activation (Gemm style)."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(n_in, n_hidden)).astype(np.float32) * 0.5
    b1 = rng.normal(size=(n_hidden,)).astype(np.float32) * 0.1
    W2 = rng.normal(size=(n_hidden, n_out)).astype(np.float32) * 0.5
    b2 = rng.normal(size=(n_out,)).astype(np.float32) * 0.1

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_in])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n_out])

    init_W1 = helper.make_tensor("W1", TensorProto.FLOAT, W1.T.shape, W1.T.flatten())
    init_b1 = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1)
    init_W2 = helper.make_tensor("W2", TensorProto.FLOAT, W2.T.shape, W2.T.flatten())
    init_b2 = helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2)

    gemm1 = helper.make_node("Gemm", ["X", "W1", "b1"], ["H1"], transB=1)
    act1 = helper.make_node(activation_op, ["H1"], ["A1"])
    gemm2 = helper.make_node("Gemm", ["A1", "W2", "b2"], ["Y"], transB=1)

    graph = helper.make_graph(
        [gemm1, act1, gemm2],
        "MLP",
        [X],
        [Y],
        [init_W1, init_b1, init_W2, init_b2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _onnxruntime_predict(model, X):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]


def _check_onnx_activation(activation_op):
    model = _build_onnx_mlp(activation_op)
    X = np.array([[0.5, -0.3, 1.0], [-1.0, 0.2, 0.8]], dtype=np.float32)
    expected = _onnxruntime_predict(model, X)

    with (
        gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
        gp.Model(env=env) as gpm,
    ):
        x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
        pc = add_predictor_constr(gpm, model, x)
        gpm.optimize()
        np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
class TestONNXSigmoid:
    def test_sigmoid_gemm(self):
        _check_onnx_activation("Sigmoid")

    def test_sigmoid_single_sample(self):
        model = _build_onnx_mlp("Sigmoid")
        X = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        expected = _onnxruntime_predict(model, X)
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
            pc = add_predictor_constr(gpm, model, x)
            gpm.optimize()
            np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_TANH, reason="Requires Gurobi 13.0+ with tanh support")
class TestONNXTanh:
    def test_tanh_gemm(self):
        _check_onnx_activation("Tanh")

    def test_tanh_single_sample(self):
        model = _build_onnx_mlp("Tanh")
        X = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        expected = _onnxruntime_predict(model, X)
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
            pc = add_predictor_constr(gpm, model, x)
            gpm.optimize()
            np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)
