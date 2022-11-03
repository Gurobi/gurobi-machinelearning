import unittest


class TestNoDependencies(unittest.TestCase):
    """Test that we can import gurobi_ml without having
    any ML library installed"""

    def test_no_sklearn(self):  # noqa
        import gurobi_ml  # # noqa
