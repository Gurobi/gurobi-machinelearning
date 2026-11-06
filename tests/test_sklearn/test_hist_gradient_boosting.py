# Copyright © 2023-2026 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..fixed_formulation import FixedRegressionModel


class TestHistGradientBoosting(FixedRegressionModel):
    def test_monotonic_constraints(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(100, 2))
        y = 2 * X[:, 0] + np.sin(X[:, 1])
        predictor = HistGradientBoostingRegressor(
            max_iter=5,
            max_leaf_nodes=5,
            min_samples_leaf=5,
            monotonic_cst=[1, 0],
            random_state=0,
        ).fit(X, y)

        examples = np.column_stack((np.linspace(0.1, 0.9, 5), np.full(5, 0.5)))
        self.assertTrue(np.all(np.diff(predictor.predict(examples)) >= 0))
        self.fixed_model(predictor, examples, nonconvex=False)


if __name__ == "__main__":
    unittest.main()
