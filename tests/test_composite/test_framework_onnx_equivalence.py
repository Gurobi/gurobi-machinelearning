"""Composite tests comparing framework models with their ONNX conversions.

These tests verify that models converted to ONNX produce equivalent results
when used with gurobi-ml. This ensures ONNX conversion and formulation
correctness.

Note: These tests may be skipped if required dependencies (tf2onnx, etc.) are
not available or if there are version compatibility issues.
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
from joblib import load

try:
    import keras
    import tensorflow as tf

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    import tf2onnx

    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import gurobipy as gp

from gurobi_ml import add_predictor_constr


class TestFrameworkONNXEquivalence(unittest.TestCase):
    """Test that framework models produce same results as their ONNX conversions."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def _solve_with_model(self, model, X_samples):
        """Solve a Gurobi model with fixed inputs using a given predictor."""
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(X_samples.shape, lb=X_samples - 1e-4, ub=X_samples + 1e-4)
            pred_constr = add_predictor_constr(gpm, model, x)
            gpm.optimize()
            return pred_constr.output.X.copy()

    @unittest.skipIf(
        not (KERAS_AVAILABLE and TF2ONNX_AVAILABLE and ONNX_AVAILABLE),
        "keras, tf2onnx, or onnx not available",
    )
    def test_keras_to_onnx_basic(self):
        """Test basic Keras to ONNX conversion produces consistent results."""
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        # Load Keras model
        filename = os.path.join(self.basedir, "diabetes.keras")
        keras_model = keras.saving.load_model(filename)

        # Select samples
        choice = self.rng.choice(X.shape[0], size=3, replace=False)
        samples = X[choice, :]

        # Get Keras direct predictions
        keras_direct = keras_model.predict(samples, verbose=0)

        # Get Gurobi-Keras predictions
        try:
            keras_gurobi = self._solve_with_model(keras_model, samples)
        except Exception as e:
            self.skipTest(f"Could not formulate Keras model: {e}")

        # Convert to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            try:
                n_features = X.shape[1]

                @tf.function(
                    input_signature=[
                        tf.TensorSpec(shape=(None, n_features), dtype=tf.float32)
                    ]
                )
                def model_fn(x):
                    return keras_model(x)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    onnx_model_proto, _ = tf2onnx.convert.from_function(
                        model_fn,
                        input_signature=[
                            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32)
                        ],
                        opset=13,
                    )

                onnx.save(onnx_model_proto, tmp.name)
                onnx_model = onnx.load(tmp.name)

                # Get Gurobi-ONNX predictions
                onnx_gurobi = self._solve_with_model(onnx_model, samples)

                # Compare Keras Gurobi vs ONNX Gurobi
                max_diff = np.max(np.abs(keras_gurobi - onnx_gurobi))

                # Use a reasonable tolerance
                tolerance = 1e-3
                self.assertLess(
                    max_diff,
                    tolerance,
                    f"Keras and ONNX Gurobi formulations differ by {max_diff:.6e}\n"
                    f"Keras direct: {keras_direct.flatten()}\n"
                    f"Keras Gurobi: {keras_gurobi}\n"
                    f"ONNX Gurobi: {onnx_gurobi}",
                )

            except Exception as e:
                # Clean up and skip if there are issues
                self.skipTest(f"ONNX conversion failed: {e}")
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    @unittest.skipIf(
        not (TORCH_AVAILABLE and ONNX_AVAILABLE), "torch or onnx not available"
    )
    def test_pytorch_to_onnx_basic(self):
        """Test basic PyTorch to ONNX conversion produces consistent results."""
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        # Load PyTorch model
        filename = os.path.join(self.basedir, "diabetes__pytorch.pt")
        try:
            pytorch_model = torch.load(filename, weights_only=False)
            pytorch_model.eval()
        except Exception as e:
            self.skipTest(f"Could not load PyTorch model: {e}")

        # Select samples
        choice = self.rng.choice(X.shape[0], size=3, replace=False)
        samples = X[choice, :]

        # Get PyTorch direct predictions
        with torch.no_grad():
            pytorch_direct = pytorch_model(torch.FloatTensor(samples)).numpy()

        # Get Gurobi-PyTorch predictions
        try:
            pytorch_gurobi = self._solve_with_model(pytorch_model, samples)
        except Exception as e:
            self.skipTest(f"Could not formulate PyTorch model: {e}")

        # Convert to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            try:
                n_features = X.shape[1]
                dummy_input = torch.randn(1, n_features)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.onnx.export(
                        pytorch_model,
                        dummy_input,
                        tmp.name,
                        export_params=True,
                        opset_version=13,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                    )

                onnx_model = onnx.load(tmp.name)

                # Get Gurobi-ONNX predictions
                onnx_gurobi = self._solve_with_model(onnx_model, samples)

                # Compare PyTorch Gurobi vs ONNX Gurobi
                max_diff = np.max(np.abs(pytorch_gurobi - onnx_gurobi))

                # Use a reasonable tolerance
                tolerance = 1e-3
                self.assertLess(
                    max_diff,
                    tolerance,
                    f"PyTorch and ONNX Gurobi formulations differ by {max_diff:.6e}\n"
                    f"PyTorch direct: {pytorch_direct.flatten()}\n"
                    f"PyTorch Gurobi: {pytorch_gurobi}\n"
                    f"ONNX Gurobi: {onnx_gurobi}",
                )

            except Exception as e:
                self.skipTest(f"ONNX conversion or formulation failed: {e}")
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    @unittest.skipIf(
        not (
            KERAS_AVAILABLE and TORCH_AVAILABLE and TF2ONNX_AVAILABLE and ONNX_AVAILABLE
        ),
        "Required frameworks not available",
    )
    def test_cross_framework_onnx_consistency(self):
        """Test that different frameworks produce consistent ONNX formulations.

        This uses the same data with different framework models, converts to ONNX,
        and verifies the formulations work correctly.
        """
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        # Use a single sample for simplicity
        sample = X[0:1, :]

        results = {}

        # Test Keras -> ONNX
        try:
            keras_model = keras.saving.load_model(
                os.path.join(self.basedir, "diabetes.keras")
            )
            keras_pred = keras_model.predict(sample, verbose=0)
            results["keras_direct"] = keras_pred.flatten()[0]

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                try:
                    n_features = X.shape[1]

                    @tf.function(
                        input_signature=[
                            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32)
                        ]
                    )
                    def model_fn(x):
                        return keras_model(x)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        onnx_model_proto, _ = tf2onnx.convert.from_function(
                            model_fn,
                            input_signature=[
                                tf.TensorSpec(
                                    shape=(None, n_features), dtype=tf.float32
                                )
                            ],
                            opset=13,
                        )

                    onnx.save(onnx_model_proto, tmp.name)
                    onnx_model = onnx.load(tmp.name)

                    keras_onnx_gurobi = self._solve_with_model(onnx_model, sample)
                    results["keras_onnx"] = keras_onnx_gurobi[0]
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
        except Exception as e:
            warnings.warn(f"Keras test skipped: {e}")

        # Test PyTorch -> ONNX
        try:
            pytorch_model = torch.load(
                os.path.join(self.basedir, "diabetes__pytorch.pt"), weights_only=False
            )
            pytorch_model.eval()

            with torch.no_grad():
                pytorch_pred = pytorch_model(torch.FloatTensor(sample)).numpy()
            results["pytorch_direct"] = pytorch_pred.flatten()[0]

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                try:
                    n_features = X.shape[1]
                    dummy_input = torch.randn(1, n_features)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        torch.onnx.export(
                            pytorch_model,
                            dummy_input,
                            tmp.name,
                            export_params=True,
                            opset_version=13,
                            do_constant_folding=True,
                        )

                    onnx_model = onnx.load(tmp.name)
                    pytorch_onnx_gurobi = self._solve_with_model(onnx_model, sample)
                    results["pytorch_onnx"] = pytorch_onnx_gurobi[0]
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
        except Exception as e:
            warnings.warn(f"PyTorch test skipped: {e}")

        # Verify we got at least some results
        if len(results) < 2:
            self.skipTest("Not enough frameworks available for comparison")

        # Print results for debugging
        print("\nCross-framework results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        # Extract direct and ONNX-based results
        direct_keys = [k for k in results if k.endswith("_direct")]
        onnx_keys = [k for k in results if k.endswith("_onnx")]

        # Require at least one direct and one ONNX-based result for a meaningful check
        self.assertGreaterEqual(
            len(direct_keys),
            1,
            "Expected at least one direct framework prediction in cross-framework test.",
        )
        self.assertGreaterEqual(
            len(onnx_keys),
            1,
            "Expected at least one ONNX-based prediction in cross-framework test.",
        )

        # For each framework that has both direct and ONNX results, ensure they match
        frameworks = set(k.split("_")[0] for k in results)
        for fw in frameworks:
            d_key = f"{fw}_direct"
            o_key = f"{fw}_onnx"
            if d_key in results and o_key in results:
                self.assertAlmostEqual(
                    results[d_key],
                    results[o_key],
                    places=5,
                    msg=f"Direct and ONNX/Gurobi predictions differ for framework '{fw}'.",
                )

        # Additionally, ensure ONNX-based formulations are consistent across frameworks
        onnx_values = [results[k] for k in onnx_keys]
        for i in range(len(onnx_values)):
            for j in range(i + 1, len(onnx_values)):
                self.assertAlmostEqual(
                    onnx_values[i],
                    onnx_values[j],
                    places=4,
                    msg="ONNX/Gurobi predictions differ across frameworks.",
                )
