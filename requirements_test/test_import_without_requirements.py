import unittest


class TestNoRequirement(unittest.TestCase):
    """Test that we can import gurobi_ml without having
    any ML library installed"""

    def test_no_sklearn(self):  # noqa
        with self.assertRaises(ModuleNotFoundError):
            import sklearn  # noqa
        with self.assertRaises(ModuleNotFoundError):
            import tensorflow  # noqa
        with self.assertRaises(ModuleNotFoundError):
            import torch  # noqa
        import gurobi_ml  # # noqa
