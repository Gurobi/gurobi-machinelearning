"""FixedRegressionModel-style tests for smooth activation neural networks.

Mirrors test_pytorch_formulations.py but for smooth (nonlinear) activations:
nn.Sigmoid, nn.Tanh, nn.Softplus.  Each test fixes the input to a Gurobi model,
embeds the network, solves, and verifies the output matches PyTorch's forward pass.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn
import requests
from functools import lru_cache

from ..fixed_formulation import FixedRegressionModel
from gurobi_ml._grb_version import HAS_NLFUNC, HAS_TANH


@lru_cache(maxsize=1)
def fetch_diabetes_data(url):
    """Fetch raw bytes of the diabetes dataset with error handling and caching."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch diabetes dataset: {e}")


def _make_smooth_mlp(activation_cls, n_in, hidden=8, seed=0):
    """Build a tiny smooth MLP with deterministic random weights."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        activation_cls(),
        nn.Linear(hidden, hidden),
        activation_cls(),
        nn.Linear(hidden, 1),
    )


def _case(activation_cls, n_in, seed=0):
    model = _make_smooth_mlp(activation_cls, n_in, seed=seed)
    model.eval()
    return {"predictor": model, "nonconvex": 1}


@unittest.skipUnless(HAS_NLFUNC, "Requires Gurobi 12.0+ with nlfunc support")
class TestSmoothActivationFormulations(FixedRegressionModel):
    """Verify smooth-activation network embeddings match PyTorch forward pass."""

    def load_diabetes(self):
        url = "https://github.com/scikit-learn/scikit-learn/raw/refs/heads/main/sklearn/datasets/data/diabetes_data_raw.csv.gz"
        try:
            from io import BytesIO
            import gzip

            raw = fetch_diabetes_data(url)
            try:
                with gzip.open(BytesIO(raw), "rt") as f:
                    diabetes_data = np.loadtxt(f)
            except gzip.BadGzipFile:
                diabetes_data = np.loadtxt(BytesIO(raw))
            x = diabetes_data[:, :-1]
            # Normalize to zero mean and unit std to keep values numerically stable
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            return x
        except Exception as e:
            raise RuntimeError(f"Error loading diabetes dataset: {e}")

    # ------------------------------------------------------------------ sigmoid
    def test_diabetes_sigmoid(self):
        """Sigmoid network on diabetes data."""
        x = self.load_diabetes()
        case = _case(nn.Sigmoid, x.shape[1], seed=1)
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")

    def test_diabetes_sigmoid_wide(self):
        """Wider network (16 neurons) with sigmoid."""
        x = self.load_diabetes()
        torch.manual_seed(2)
        model = nn.Sequential(nn.Linear(x.shape[1], 16), nn.Sigmoid(), nn.Linear(16, 1))
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")

    # -------------------------------------------------------------------- tanh
    @unittest.skipUnless(HAS_TANH, "Requires Gurobi 13.0+ with tanh support")
    def test_diabetes_tanh(self):
        x = self.load_diabetes()
        case = _case(nn.Tanh, x.shape[1], seed=3)
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")

    @unittest.skipUnless(HAS_TANH, "Requires Gurobi 13.0+ with tanh support")
    def test_diabetes_tanh_wide(self):
        """Wider network (16 neurons) with tanh."""
        x = self.load_diabetes()
        torch.manual_seed(4)
        model = nn.Sequential(nn.Linear(x.shape[1], 16), nn.Tanh(), nn.Linear(16, 1))
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")

    # ---------------------------------------------------------------- softplus
    def test_diabetes_softplus(self):
        x = self.load_diabetes()
        case = _case(nn.Softplus, x.shape[1], seed=5)
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")

    def test_diabetes_softplus_beta2(self):
        """Softplus with non-default beta=2."""
        x = self.load_diabetes()
        torch.manual_seed(6)
        model = nn.Sequential(
            nn.Linear(x.shape[1], 8),
            nn.Softplus(beta=2, threshold=20),
            nn.Linear(8, 1),
        )
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")

    # --------------------------------------------------------- mixed activations
    @unittest.skipUnless(HAS_TANH, "Requires Gurobi 13.0+ with tanh support")
    def test_diabetes_sigmoid_tanh_mixed(self):
        """Network with sigmoid in first layer and tanh in second."""
        x = self.load_diabetes()
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Linear(x.shape[1], 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, x, 5, "all")
        self.do_one_case(case, x, 6, "pairs")
