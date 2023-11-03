Mixed Integer Formulations
##########################

In this page, we give a quick overview of the mixed-integer formulations used to
represent the various regression models supported by the package.

Our goal is in particular to highlight the cases where the formulation are not
exact and how to deal with potential errors in the solution. This applies in
particular to our models for the logistic regression and decision trees (also
random forest and gradient boosting that are based on decision trees)

Throughout,
we denote by :math:`x` the input of the regression (i.e. the independent variables)
and :math:`y` the output of the regression model (i.e. the dependent variables).


Linear Regression
=================

Denoting by :math:`\beta \in \mathbb R^{p+1}` the computed weights of linear regression,
its model takes the form

.. math::

  y = \sum_{i=1}^p \beta_i x_i + \beta_0.

Since this is linear, it can be represented directly in Gurobi using
linear constraints. Note that the model fits other techniques than ordinary linear
regression such as Ridge or Lasso.

Logistic Regression
===================

Denoting by :math:`f(x) = \frac{1}{1 - e^{-x}}` the standard logistic function
and with the same notations as above, the model for logistic regression reads

.. math::

  y = f(\sum_{i=1}^p \beta_i x_i + \beta_0) = \frac{1}{1 - e^{- \sum_{i=1}^p
  \beta_i x_i - \beta_0}}

This model is formulated in Gurobi by using the logistic
`general function
constraint
<https://www.gurobi.com/documentation/current/refman/constraints.html#subsubsection:GenConstrFunction>`_.
First an intermediate free variable :math:`\omega = \sum_{i=1}^p \beta_i x_i +
\beta_0` is created, and then we can express :math:`y = f(\omega)` using the
general constraint.

With version 11, Gurobi introduced direct algorithmic support of nonlinear functions.
We enable it by setting the attribute `FuncNonLinear` to 1 for the logistic functions
created by Gurobi Machine Learning.

Older versions of Gurobi make a piecewise linear approximation of the logistic
function. By default, the approximation guarantees a maximal error of
:math:`10^{-2}`. Those parameters can be tuned by setting the `pwl_attributes`
keyword argument when the constraints is added.


Neural Networks
===============

The package currently models dense neural network with ReLU activations. For a
given neuron the relation between its inputs and outputs is given by:

.. math::

    y = \max(\sum_{i=1}^p \beta_i x_i + \beta_0, 0).

The relationship is formulated in the optimization model by using Gurobi
:math:`max` `general constraint
<https://www.gurobi.com/documentation/latest/refman/constraints.html#subsubsection:GeneralConstraints>`_
with:

.. math::

    & \omega = \sum_{i=1}^p \beta_i x_i + \beta_0

    & y = \max(\omega, 0)

with :math:`\omega` an auxiliary free variable. The neurons are then connected
according to the topology of the network.


Decision Tree Regression
========================

In a decision tree, each leaf :math:`l` is defined by a number of constraints
on the input features of the tree that correspond to the branches taken in the
path leading to :math:`l`. For a node :math:`v`, we denote by :math:`i_v` the
feature used for splitting and by :math:`\theta_v` the value at which the split
is made. At a leaf :math:`l` of the tree, we have a set :math:`\mathcal L_l` of inequalities of
the form :math:`x_{i_v} \le \theta_v` corresponding to the left branches leading to
:math:`l` and a set :math:`\mathcal R_l` of inequalities of
the form :math:`x_{i_v} > \theta_v` corresponding to the right branches.

We formulate decision trees by introducing one binary decision variable
:math:`\delta_l` for each leaf of the tree (and each input vector).

We introduce the constraint

.. math::
   \sum_{l} \delta_l = 1,

imposing that at least one leaf is chosen.

Then for each leaf, the inequalities describing :math:`\mathcal L_l` and :math:`\mathcal R_l`
are imposed using indicator constraints:

.. math::

   & \delta_l = 1 \rightarrow x_{i_v} \le \theta_v, & & \text{for } x_{i_v} \le \theta_v \in \mathcal L_l,

   & \delta_l = 1 \rightarrow x_{i_v} \ge \theta_v + \epsilon, & & \text{for } x_{i_v} > \theta_v \in \mathcal R_l.

A difficulty here is that the strictly greater than constraints of :math:`\mathcal R_l`
can't be represented exactly in a mixed integer optimization model. To
approximate it, we introduce a small threshold :math:`\epsilon`. We discuss
below the trade-offs for choosing a value for :math:`\epsilon`.

In our implementation, :math:`\epsilon` can be specified by a keyword parameter
of :func:`add_decision_tree_regressor_constr <gurobi_ml.sklearn.add_decision_tree_regressor_constr>`. The default
value for :math:`\epsilon` is 0. This means in particular that if :math:`x_{i_v}
= \theta_v` in the solution, the model doesn't discriminate between the two child
nodes of :math:`v` and either direction may be picked. This may also
happen whenever :math:`\epsilon` is set to a value that is below the
`feasibility tolerance
<https://www.gurobi.com/documentation/current/refman/feasibilitytol.html#parameter:FeasibilityTol>`_
of Gurobi. If the value is instead set above the feasibility tolerance, then the
left and right nodes are correctly discriminated by the model, but a small
interval is created between :math:`\theta` and :math:`\theta + \epsilon` where
there is no feasible solution. This may artificially make the optimization model infeasible
depending on how tightly the input of the decision tree regressor is
constrained.

The reasoning behind our default setting is that even though there may be a
difference between the output value of the Gurobi model and the prediction of
the original decision tree, it only corresponds to a small perturbation in the
values of the input variables.

Random Forest Regression
========================

The regression model of Random Forests is a linear combination of decision trees.
Each decision tree is represented using the model above. The same difficulties
with the choice of :math:`\epsilon` apply to this case.

We note additionally that the random forests are often very large and generating
their representation in Gurobi may take a significant amount of time.

Gradient Boosting Regression
============================

The gradient boosting regressor is a linear combination of decision trees. Each
decision tree is represented using the model above. The same difficulties with
the choice of :math:`\epsilon` apply to this case.

We note additionally that the gradient boosting regressors are often very large
and generating their representation in Gurobi may take a significant amount of
time.
