import unittest

import gurobipy as gp
import numpy as np

from gurobi_ml import add_predictor_constr, register_predictor_constr
from gurobi_ml.modeling.neuralnet import BaseNNConstr


class MyNNConstr(BaseNNConstr):
    """Predict a Gurobi matrix variable using a neural network that
    takes another Gurobi matrix variable as input.
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        clean_predictor=False,
        **kwargs,
    ):
        BaseNNConstr.__init__(
            self,
            gp_model,
            predictor,
            input_vars,
            output_vars,
            clean_predictor=clean_predictor,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi"""
        neural_net = self.predictor
        n_layers = neural_net["n_layers"]
        intercepts = neural_net["intercepts"]
        coefs = neural_net["coefs"]

        activation = self.act_dict["relu"]

        input_vars = self._input
        output = None

        for i in range(n_layers):
            layer_coefs = coefs[i]
            layer_intercept = intercepts[i]

            # For last layer change activation
            if i == n_layers - 1:
                activation = self.act_dict["identity"]
                output = self._output

            layer = self.add_dense_layer(
                input_vars,
                layer_coefs,
                layer_intercept,
                activation,
                output,
                name=f"layer{i}",
            )
            input_vars = layer._output  # pylint: disable=W0212
            self._gp_model.update()
        assert self._output is not None


def abs_model(X, y, nn, inf_bound, registered):
    bound = 100
    with gp.Model() as model:
        samples, dim = X.shape
        assert samples == y.shape[0]
        # Decision variables
        beta = model.addMVar(dim, lb=-bound, ub=bound, name="beta")  # Weights
        diff = model.addMVar((samples, 1), lb=-inf_bound, ub=inf_bound, name="diff")
        abs_diff = model.addMVar((samples, 1), lb=-inf_bound, ub=inf_bound, name="absdiff")

        model.addConstr(X @ beta - y == diff[:, 0])
        model.setObjective(abs_diff.sum(), gp.GRB.MINIMIZE)

        model.update()

        if nn:
            if registered:
                add_predictor_constr(model, nn, diff, abs_diff)
            else:
                MyNNConstr(model, nn, diff, abs_diff)
        else:
            for i in range(samples):
                model.addConstr(abs_diff[i, 0].item() == gp.abs_(diff[i, 0].item()))

        model.Params.OutputFlag = 0
        model.Params.WorkLimit = 100

        model.optimize()
        return model.ObjVal


class TestNNFormulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test

        I have absolutely no clue how I came up with this model and this data"""

        cls.y = np.array(
            [
                21.31446068,
                36.35152748,
                -161.12971621,
                166.77466327,
                -51.14555201,
                43.60880995,
                67.87833908,
                -14.82619044,
                113.72780017,
                -92.49550518,
                -125.65651202,
                59.69787755,
                175.01554444,
                -42.45127666,
                -140.14088357,
                21.79251352,
                -190.29125769,
                -232.07847741,
                105.7387485,
                245.56045298,
                131.34665771,
                168.87216422,
                -32.5771315,
                300.32075565,
                -194.3924965,
                -217.55498011,
                -139.90646417,
                -105.24615409,
                -169.96987143,
                80.56176677,
            ]
        )

        cls.X = np.array(
            (
                [
                    [0.06651722, 0.42833187, -0.02818223],
                    [1.05445173, -1.07075262, -0.17992484],
                    [0.15634897, -0.34791215, -1.98079647],
                    [1.45427351, 0.14404357, 0.4105985],
                    [-0.88778575, 0.37816252, 0.15494743],
                    [0.12898291, 0.72909056, 0.0519454],
                    [-0.20515826, 1.49407907, 0.33367433],
                    [0.77749036, -1.25279536, -0.4380743],
                    [-0.38732682, 1.20237985, 1.23029068],
                    [-1.18063218, -0.51080514, 0.3869025],
                    [-1.16514984, 0.05616534, -0.31155253],
                    [-0.10321885, -0.15135721, 0.95008842],
                    [1.46935877, 1.53277921, -0.18718385],
                    [-1.53624369, 0.46566244, 0.90082649],
                    [-0.81314628, -0.35955316, -0.67246045],
                    [-0.74216502, 0.8644362, 0.6536186],
                    [-1.42001794, -1.04855297, -0.30230275],
                    [-2.55298982, -0.85409574, 0.3130677],
                    [0.44386323, 0.12167502, 0.76103773],
                    [0.97873798, 0.40015721, 1.76405235],
                    [0.04575852, -1.45436567, 2.26975462],
                    [-0.97727788, 1.86755799, 2.2408932],
                    [-0.36274117, -0.63432209, 0.3024719],
                    [1.17877957, 1.89588918, 1.48825219],
                    [-0.90729836, 0.46278226, -1.63019835],
                    [-0.89546656, -0.21274028, -1.61389785],
                    [-0.57884966, -0.87079715, -0.68481009],
                    [-0.50965218, 1.9507754, -1.70627019],
                    [-0.40178094, 0.17742614, -1.7262826],
                    [0.40234164, -1.23482582, 1.13940068],
                ]
            )
        )
        cls.nn = {
            "intercepts": [np.array([0, 0]), np.array([0])],
            "coefs": [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])],
            "n_layers": 2,
        }

    def do_the_test(self, bound, registered=False):
        """Test the network for absolute value with relu formulation"""
        val1 = abs_model(self.X, self.y, False, bound, registered)
        val2 = abs_model(self.X, self.y, self.nn, bound, registered)
        self.assertAlmostEqual(val1, val2, places=4)

    def test_no_bounds(self):
        self.do_the_test(gp.GRB.INFINITY)

    def test_bounds(self):
        self.do_the_test(100)

    def test_register_no_bounds(self):
        register_predictor_constr(dict, MyNNConstr)
        self.do_the_test(gp.GRB.INFINITY, True)

    def test_register_bounds(self):
        register_predictor_constr(dict, MyNNConstr)
        self.do_the_test(100, True)
