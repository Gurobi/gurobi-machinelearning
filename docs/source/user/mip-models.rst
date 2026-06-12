Mixed Integer Formulations
##########################

In this page, we give a quick overview of the mixed-integer formulations used to
represent the various regression models supported by the package.

Our goal is in particular to highlight the cases where the formulations are not
exact and how to deal with potential errors in the solution. This applies in
particular to our models for logistic regression and decision trees (also
random forest and gradient boosting that are based on decision trees).

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

Denoting by :math:`f(x) = \frac{1}{1 + e^{-x}}` the standard logistic function
and with the same notations as above, the model for logistic regression reads

.. math::

  y = f(\sum_{i=1}^p \beta_i x_i + \beta_0) = \frac{1}{1 + e^{- \sum_{i=1}^p
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
keyword argument when the constraint is added.


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

In a decision tree, each leaf :math:`l` is defined by a set of constraints
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

imposing that exactly one leaf is chosen.

Then for each leaf, the inequalities describing :math:`\mathcal L_l` and :math:`\mathcal R_l`
are imposed using indicator constraints:

.. math::
   :nowrap:

   \begin{align*}
   & \delta_l = 1 \rightarrow x_{i_v} \le \theta_v, & & \text{for } x_{i_v} \le \theta_v \in \mathcal L_l,\\
   & \delta_l = 1 \rightarrow x_{i_v} \ge \theta_v + \epsilon, & & \text{for } x_{i_v} > \theta_v \in \mathcal R_l.
   \end{align*}

Two numerical parameters control the accuracy of this formulation, both
exposed as keyword arguments of
:func:`add_decision_tree_regressor_constr <gurobi_ml.sklearn.add_decision_tree_regressor_constr>`.

**epsilon** (:math:`\epsilon`, default 0) approximates the strict inequality in
:math:`\mathcal R_l`. For :math:`\epsilon` to correctly discriminate left and
right branches it must exceed Gurobi's
:external+gurobi:ref:`FeasibilityTol <parameterfeasibilitytol>`
(default :math:`10^{-6}`); below that threshold the solver treats
:math:`x_{i_v} \ge \theta_v + \epsilon` and :math:`x_{i_v} \ge \theta_v`
as equivalent. Setting :math:`\epsilon` above the feasibility tolerance does
enforce the correct branch, but creates a gap :math:`[\theta_v,\,\theta_v + \epsilon]`
with no feasible solution, which may make the model infeasible when the inputs
are tightly constrained. The default of 0 avoids this, at the cost of
ambiguity exactly at a split boundary.

**safety_floor** (default 0, i.e. disabled) addresses a different issue: when
:math:`|\theta_v|` itself is smaller than
:external+gurobi:ref:`FeasibilityTol <parameterfeasibilitytol>`,
the solver treats 0 and :math:`\theta_v` as equal and the indicator constraints become
ineffective. Setting ``safety_floor`` clamps those thresholds to
:math:`\pm\,\text{safety\_floor}`, which fixes the issue provided
``safety_floor`` :math:`\ge` ``FeasibilityTol``. Because the clamping shifts
decision boundaries it can distort models whose legitimate thresholds are
genuinely near zero, so the parameter is opt-in.

Random Forest Regression
========================

The regression model of Random Forests is a linear combination of decision trees.
Each decision tree is represented using the model above. The same difficulties
with the choice of :math:`\epsilon` and ``safety_floor`` apply to this case.

We note additionally that the random forests are often very large and generating
their representation in Gurobi may take a significant amount of time.

Gradient Boosting Regression
============================

The gradient boosting regressor is a linear combination of decision trees. Each
decision tree is represented using the model above. The same difficulties with
the choice of :math:`\epsilon` and ``safety_floor`` apply to this case.

We note additionally that the gradient boosting regressors are often very large
and generating their representation in Gurobi may take a significant amount of
time.
