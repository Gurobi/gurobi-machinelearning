Mixed Integer Formulations
##########################

In this page, we give a quick overview of the mixed-integer formulations used to
represent the various regression models supported by the package.

Our goal is in particular to highlight the cases where the formulation are not
exact and how to deal with potential errors in the solution. This applies in
particular to our models for the logistic regression and decision trees (also
random forest and gradient boosting that are based on decision trees)

Throughout,
we denote by :math:`x` the independent variables (or input of the regression
models) and :math:`y` the dependent variables (or output of the regression model)


Linear Regression
=================

With given weights :math:`\beta \in \mathbb R^{p+1}` the linear regression model
takes the form

.. math::

  y = \sum_{i=1}^p \beta_i x_i + \beta_0.

Since this is purely linear, it can be represented using linear constraints in
Gurobi. Note that the model fits other techniques than ordinary linear
regression such as Ridge or Lasso.

Logistic Regression
===================

Denoting by :math:`f(x) = \frac{1}{1 - e^{-x}}` the standard logistic function
and with the same notations as above, the model for logistic regression reads

.. math::

  y = f(\sum_{i=1}^p \beta_i x_i + \beta_0) = \frac{1}{1 - e^{- \sum_{i=1}^p
  \beta_i x_i - \beta_0}}

This model is formulated in Gurobi by using the logistic `general function
constraint
<https://www.gurobi.com/documentation/latest/refman/constraints.html#subsubsection:GenConstrFunction>`_.
First an intermediate free variable :math:`\omega = \sum_{i=1}^p \beta_i x_i +
\beta_0` is created, and then we can express :math:`y = f(\omega)` using the
general constraint.

Internally, Gurobi then makes a piecewise linear approximation of the logistic
function. By default, the approximation guarantees a maximal error of
:math:`10^{-2}`. Those parameters can be tuned by setting the _pwl_attributes_
keyword argument when the constraints are added (see
:doc:`mlm-examples/student_admission` for an example of how to change the
default values).


Neural Networks
===============

The package currently models dense neural network with ReLU activations. For a
given neuron the relation between its inputs and outputs is given by:

.. math::

    y = \max(\sum_{i=1}^p \beta_i x_i + \beta_0, 0).

The relationship is formulated in the optimization model by using the
:math:`max` `general constraint
<https://www.gurobi.com/documentation/latest/refman/constraints.html#subsubsection:GeneralConstraints>`_
with:

.. math::

    & \omega = \sum_{i=1}^p \beta_i x_i + \beta_0

    & y = \max(\omega, 0)

with :math:`\omega` an auxiliary free variable. The neurons are then connected
according to the topology of the network.


Decision Trees
==============

For representing decision tree in Gurobi, we add one binary decision variable
:math:`\delta` for each node of the tree. Those variables represent the path
taken in the decision tree with the values of the input variables. For a node
:math:`i`, we will have :math:`\delta_i = 1` if node :math:`i` is on the
decision path and :math:`\delta =0` otherwise.

Let :math:`i` be a node with left child :math:`j` and right child :math:`k`. The
connectivity of the decision path is modeled with the following constraint that
specifies that if :math:`i` is on the decision path then either :math:`j` or
:math:`k` is on the decision path:

.. math::

   \delta_j + \delta_k = \delta_i


For further detailing the formulation, we differentiate between splitting nodes
and leafs of the tree.

We first consider the leafs of the tree that are more simple. Let :math:`i` be a
leaf of the decision tree, then if :math:`i` is on the decision path, the output
value of the regression is fixed to the value :math:`\theta_i`. We model this through
the indicator constraint:

.. math::

   \delta_i = 1 \rightarrow y = \theta_i.

Note that :math:`y` here might be multidimensional.

Now we consider the more complicated case of a splitting node. Let :math:`i` be
a splitting node with left child :math:`j` and right child :math:`k`.
Furthermore, we denote by :math:`s_i` the index of the feature used for
splitting in node :math:`i` and by :math:`theta_i` the threshold value. By
definition, if node :math:`i` is on the decision path, then we proceed to node
:math:`j` if :math:`x_{s_i} \le \theta_i` and to node :math:`k` otherwise (i.e.
if :math:`x_{s_i} > \theta_i`).

A difficulty here is that the strictly greater than constraint for defining node
:math:`k` can't be represented exactly in a mixed integer optimization model. To
approximate it, we introduce a small threshold :math:`\epsilon`. We discuss
below the tradeoffs for choosing a value for :math:`\epsilon`.

Using :math:`\epsilon`, the splitting can be represented by the pairs of
indicator constraints:

.. math::

   & \delta_j = 1 \rightarrow x_{s_i} \le \theta_i,

   & \delta_k = 1 \rightarrow x_{s_i} \ge \theta_i + \epsilon.

In our implementation, :math:`\epsilon` can be specified by a keyword parameter
of :func:`add_decision_tree_regressor_constr <gurobi_ml.sklearn.add_decision_tree_regressor_constr>`. The default
value for :math:`\epsilon` is 0. This means in particular that if :math:`x_{s_i}
= \theta_i` in the solution, the model doesn't discriminate between nodes
:math:`k` and :math:`j` and either may be picked in the decision path. This may
happen also whenever :math:`\epsilon` is set to a value that is below the
`feasibility tolerance
<https://www.gurobi.com/documentation/latest/refman/feasibilitytol.html#parameter:FeasibilityTol>`_
of Gurobi. If the value is instead set above the feasibility tolerance, then the
left and right nodes are correctly discriminated by the model, but a small
interval is created between :math:`\theta` and :math:`\theta + \epsilon` where
there is no feasible solution. This might result in an infeasible model
depending on how tightly the input of the decision tree regressor is
constrained.

The reasoning behind our default setting is that even though there may be a
difference between the output value of the Gurobi model and the prediction of
the original decision tree, it mostly corresponds to a small perturbation in the
values of the input variables.

Random Forests
==============

The regression model of Random Forests is a linear combination of decision trees.
Each decision tree is represented using the model above. The same difficulties
with the choice of :math:`epsilon` apply to this case.

We note additionally that the random forests are often very large and generating
their representation in Gurobi may take a significant amount of time.

Gradient Boosting Regressors
=============================

The gradient boosting regressor is a linear combination of decision trees. Each
decision tree is represented using the model above. The same difficulties with
the choice of :math:`epsilon` apply to this case.

We note additionally that the gradient boosting regressors are often very large
and generating their representation in Gurobi may take a significant amount of
time.
