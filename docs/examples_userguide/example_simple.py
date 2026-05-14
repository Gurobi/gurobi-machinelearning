"""
Usage Example
=============

In this page, we provide a simple example of using the Gurobi Machine
Learning package.

The example is entirely abstract. Its aim is only to illustrate the
basic functionalities of the package in the most simple way. For some
more realistic applications, please refer to the notebooks in the
`examples <../auto_examples/index.html>`__ section.

Before proceeding to the example itself, we need to import a number of
packages. Here, we will use Scikit-learn to train regression models. We
generate random data for the regression using the
`make_regression <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html>`__
function. For the regression model, we use a `multi-layer perceptron
regressor <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`__
neural network. We import the corresponding objects.

"""

import gurobipy as gp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor

from gurobi_ml import add_predictor_constr

######################################################################
# Certainly, we need gurobipy to build an optimization model and from the
# gurobi_ml package we need the
# :func:`add_predictor_constr <gurobi_ml.add_predictor_constr>`.
# function. We also need numpy.
#


######################################################################
# We start by building artificial data to train our regressions. To do so,
# we use *make_regression* to obtain data with 10 features.
#

X, y = make_regression(n_features=10, noise=1.0)


######################################################################
# Now, create the *MLPRegressor* object and fit it.
#

nn = MLPRegressor(hidden_layer_sizes=[20] * 2, max_iter=10000, random_state=1)

nn.fit(X, y)


######################################################################
# We now turn to the optimization model. In the spirit of adversarial
# machine learning examples, we use some training examples. We pick
# :math:`n` training examples randomly. For each of the examples, we want
# to find an input that is in a small neighborhood of it that leads to the
# output that is closer to :math:`0` with the regression.
#
# Denoting by :math:`X^E` our set of examples and by :math:`g` the
# prediction function of our regression model, our optimization problem
# reads:
#
# .. math::
#
#
#    \begin{aligned}
#    &\min \sum_{i=1}^n y_i^2 \\
#    &\text{s.t.:}\\
#    &y_i = g(X_i) & & i = 1, \ldots, n,\\
#    &X^E - \delta \leq X \leq X^E + \delta,\\
#    \end{aligned}
#
# where :math:`X` is a matrix of variables of dimension
# :math:`n \times 10` (the number of examples we consider and number of
# features in the regression respectively), :math:`y` is a vector of free
# (unbounded) variables and :math:`\delta` a small positive constant.
#
# First, let’s pick randomly 2 training examples using numpy, and create
# our :external+gurobi:py:class:`Gurobi model<Model>`.
#

n = 2
index = np.random.choice(X.shape[0], n, replace=False)
X_examples = X[index, :]
y_examples = y[index]

m = gp.Model()


######################################################################
# Our only decision variables in this case, are the five inputs and
# outputs for the regression. We use :external+gurobi:py:class:`MVar` matrix variables
# that are most convenient in this case.
#
# The input variables have the same shape as ``X_examples``. Their lower
# bound is ``X_examples - delta`` and their upper bound
# ``X_examples + delta``.
#
# The output variables have the shape of ``y_examples`` and are unbounded.
# By default, in Gurobi variables are non-negative, we therefore need to
# set an infinite lower bound.
#

input_vars = m.addMVar(X_examples.shape, lb=X_examples - 0.2, ub=X_examples + 0.2)
output_vars = m.addMVar(y_examples.shape, lb=-gp.GRB.INFINITY)


######################################################################
# The constraints linking ``input_vars`` and ``output_vars`` can now be
# added with the function
# :func:`add_predictor_constr <gurobi_ml.add_predictor_constr>`.
#
# Note that because of the shape of the variables this will add the 5
# different constraints.
#
# The function returns an instance of a :class:`modeling
# object <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr>`
# that we can use later on.
#

pred_constr = add_predictor_constr(m, nn, input_vars, output_vars)


######################################################################
# The method
# :func:`print_stats <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.print_stats>`
# of the modeling object outputs the details of the regression model that
# was added to the Gurobi.
#

pred_constr.print_stats()


######################################################################
# To finish the model, we set the objective, and then we can optimize it.
#

m.setObjective(output_vars @ output_vars, gp.GRB.MINIMIZE)

m.optimize()


######################################################################
# The method
# :func:`get_error <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.get_error>`
# is useful to check that the solution computed by Gurobi is correct with
# respect to the regression model we use.
#
# Let :math:`(\bar X, \bar y)` be the values of the input and output
# variables in the computed solution. The function returns
# :math:`g(\bar X) - y` using the original regression object.
#
# Normally, all values should be small and below Gurobi’s tolerances in this example.
#

pred_constr.get_error()


######################################################################
# We can look at the computed values for the output variables and compare
# them with the original target values.
#

print("Computed values")
print(pred_constr.output_values.flatten())

######################################################################

print("Original values")
print(y_examples)


######################################################################
# Finally, we can remove ``pred_constr`` with the method
# :func:`remove() <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.remove>`.
#

pred_constr.remove()
