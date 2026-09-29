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

The trees of a random forest can also be formulated as one ensemble with one of
the experimental formulations described in :ref:`Tree Ensemble Formulations`.

Gradient Boosting Regression
============================

The gradient boosting regressor is a linear combination of decision trees. Each
decision tree is represented using the model above. The same difficulties with
the choice of :math:`\epsilon` and ``safety_floor`` apply to this case.

We note additionally that the gradient boosting regressors are often very large
and generating their representation in Gurobi may take a significant amount of
time.

The trees of a gradient boosting regressor (scikit-learn, XGBoost or LightGBM)
can also be formulated as one ensemble with one of the experimental
formulations described in :ref:`Tree Ensemble Formulations`.

Tree Ensemble Formulations
==========================

.. warning::

   The formulations in this section are **experimental**. Their names,
   keyword arguments and the structure of the variables they create may
   change in future releases without a deprecation period. The default
   ``"leaf"`` formulation described above is not affected.

The formulation above models every tree of an ensemble on its own: the trees
only interact through the input variables :math:`x`. Three formulations from
the literature are available as alternatives.
``"misic"``, ``"misic_lazy"`` and ``"ocean"`` model the ensemble as a whole,
with variables shared by all trees; ``"biggs_perakis"`` models each tree on
its own, like ``"leaf"``, with a tighter linear relaxation. They are selected
with the ``formulation`` keyword argument of
:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>`:

.. code-block:: python

   pred_constr = add_predictor_constr(
       gp_model, gbt, input_vars, output_vars, formulation="misic"
   )

.. list-table::
   :header-rows: 1

   * - ``formulation``
     - Reference
   * - ``"leaf"`` (default)
     - The formulation of :ref:`Decision Tree Regression`
   * - ``"misic"``
     - :cite:t:`Misic2020`
   * - ``"misic_lazy"``
     - ``"misic"`` with lazy split constraints (see
       :ref:`Lazy Split Constraints`)
   * - ``"ocean"``
     - :cite:t:`ParmentierVidal2021`, as in their OCEAN implementation
   * - ``"biggs_perakis"``
     - :cite:t:`BiggsPerakis2025`, projected formulation

They are available for the scikit-learn
:external+sklearn:py:class:`DecisionTreeRegressor <sklearn.tree.DecisionTreeRegressor>`,
:external+sklearn:py:class:`RandomForestRegressor <sklearn.ensemble.RandomForestRegressor>` and
:external+sklearn:py:class:`GradientBoostingRegressor <sklearn.ensemble.GradientBoostingRegressor>`,
and for the XGBoost and LightGBM gradient boosting regressors (a single
decision tree is an ensemble of one tree). All formulations represent exactly
the same function; they differ in their size, in the strength of their
relaxation and in how they behave at a split threshold.

The ensemble predicts

.. math::

   y = c + \sum_{t=1}^{T} w_t f_t(x),

where :math:`f_t(x)` is the value :math:`\text{val}_l` of the leaf :math:`l`
of tree :math:`t` that :math:`x` reaches, :math:`w_t` is the weight of the tree
(e.g. the learning rate of gradient boosting) and :math:`c` a constant. For a
split node :math:`s`, :math:`\text{left}(s)` and :math:`\text{right}(s)` are the
leaves below its left and right child. For each feature :math:`i`,
:math:`\theta_{i,1} < \theta_{i,2} < \dots < \theta_{i,m_i}` are the distinct
thresholds at which *any* tree of the ensemble splits on :math:`i`.

All formulations accept the ``epsilon`` and ``safety_floor`` keyword
arguments of :ref:`Decision Tree Regression`; how the scope of ``epsilon``
differs between them is described in :ref:`Split Thresholds and Exactness`.

Mišić (``"misic"``)
-------------------

The trees share one binary variable per feature and distinct threshold of the
ensemble:

.. math::

   z_{i,j} = 1 \iff x_i \le \theta_{i,j}.

The variables of one feature are ordered and linked to the input variables by
indicator constraints:

.. math::
   :nowrap:

   \begin{align*}
   & z_{i,j} \le z_{i,j+1}, \\
   & z_{i,j} = 1 \rightarrow x_i \le \theta_{i,j}, \\
   & z_{i,j} = 0 \rightarrow x_i \ge \theta_{i,j} + \epsilon.
   \end{align*}

Every tree refers to the same variables. The number of binary variables
therefore grows with the number of distinct thresholds and not with the number
of trees, and branching on one :math:`z_{i,j}` decides the corresponding split
in all trees at once.

Each tree has one continuous variable :math:`\lambda_l \in [0, 1]` per leaf
with

.. math::

   \sum_{l} \lambda_l = 1,

and each split node :math:`s` on feature :math:`i` at threshold
:math:`\theta_{i,j}` links the leaves of its two subtrees to the shared split
variable:

.. math::

   \sum_{l \in \text{left}(s)} \lambda_l \le z_{i,j}, \qquad
   \sum_{l \in \text{right}(s)} \lambda_l \le 1 - z_{i,j}.

The output of the tree is :math:`\sum_l \text{val}_l\, \lambda_l`.

OCEAN (``"ocean"``)
-------------------

This is the formulation of :cite:t:`ParmentierVidal2021`, as implemented in
their OCEAN software. Each tree routes one unit of flow from its root to a
leaf, with one continuous flow variable :math:`\phi_v \in [0, 1]` per node:

.. math::

   \phi_{\text{root}} = 1, \qquad
   \phi_v = \phi_{\text{left child}(v)} + \phi_{\text{right child}(v)}.

One binary variable :math:`b_d` per tree and depth level :math:`d` chooses the
direction taken at that level:

.. math::

   \phi_{\text{left child}(v)} \le b_{d(v)}, \qquad
   \phi_{\text{right child}(v)} \le 1 - b_{d(v)}.

The trees share continuous feature variables. For each feature :math:`i`, the
thresholds split the domain :math:`[\ell_i, u_i]` of :math:`x_i` into intervals
of widths :math:`\Delta_{i,j}`, and :math:`\mu_{i,j} \in [0, 1]` is the
fraction of interval :math:`j` below :math:`x_i`:

.. math::

   \mu_{i,j} \ge \mu_{i,j+1}, \qquad
   x_i = \ell_i + \sum_j \Delta_{i,j}\, \mu_{i,j}.

A flow going right at threshold :math:`\theta_{i,j}` requires the interval
ending at :math:`\theta_{i,j}` to be crossed completely, a flow going left
forbids entering the interval starting at :math:`\theta_{i,j}`:

.. math::

   \phi_{\text{right child}(v)} \le \mu_{i,j}, \qquad
   \phi_{\text{left child}(v)} \le 1 - \mu_{i,j+1}.

The output of the tree is the sum of the leaf values weighted by the leaf
flows. With a positive :math:`\epsilon` the band
:math:`(\theta_{i,j}, \theta_{i,j} + \epsilon)` becomes an interval of its
own, and going right requires crossing it. The only binary variables are the
per-level branching variables :math:`b_d`, and the formulation has no
indicator constraints.

Biggs–Perakis (``"biggs_perakis"``)
------------------------------------

Each tree has one binary variable :math:`\delta_l` per leaf with
:math:`\sum_l \delta_l = 1`. Denoting by :math:`[\text{lb}_{l,i},
\text{ub}_{l,i}]` the box of inputs reaching leaf :math:`l` (clipped to the
bounds of the input variables), each feature :math:`i` the tree splits on gets
two constraints:

.. math::

   \sum_l \text{lb}_{l,i}\, \delta_l \;\le\; x_i \;\le\; \sum_l \text{ub}_{l,i}\, \delta_l.

The linear relaxation of each tree is the convex hull of the union of its leaf
boxes (projected on :math:`x`). The formulation has no shared variables: as in
the ``"leaf"`` formulation, the trees only interact through :math:`x`.

Summary
-------

.. list-table::
   :header-rows: 1

   * - ``formulation``
     - Binary variables
     - Continuous variables per tree
     - Indicator constraints
     - Finite input bounds required
   * - ``"leaf"``
     - one per leaf
     - output of the tree
     - one per leaf, plus up to two per leaf and feature
     - no
   * - ``"misic"``, ``"misic_lazy"``
     - one per distinct threshold
     - one per leaf
     - two per distinct threshold
     - no
   * - ``"ocean"``
     - one per tree and depth level
     - one per node, plus the shared :math:`\mu`
     - none
     - yes
   * - ``"biggs_perakis"``
     - one per leaf
     - output of the tree
     - none
     - yes

The ``"ocean"`` and ``"biggs_perakis"`` formulations use the bounds of the input
variables as constraint coefficients. They raise a :py:exc:`ValueError` if a
feature used in a split does not have a finite lower and upper bound. This
typically rules them out for trees placed after a transformation in a
pipeline, whose inputs are unbounded.

Calling ``print_stats()`` on the returned object shows the size of the
formulation. For ``"leaf"`` and ``"biggs_perakis"`` it also lists each tree; the
formulations with shared variables are built for the whole ensemble at once,
and only the totals are shown.

Split Thresholds and Exactness
------------------------------

With :math:`\epsilon = 0`, an input :math:`x_i` equal to a threshold
:math:`\theta_{i,j}` satisfies the constraints of both branches, and the
solver may take either one. Tree ensembles often reuse the same threshold in
many trees, and optimal solutions very often lie exactly on thresholds. The
formulations differ in what happens then:

- In ``"misic"`` and ``"misic_lazy"`` the direction is decided once, by
  the shared variable :math:`z_{i,j}`, and every tree takes the same branch.
  The chosen leaves always contain a common input, so the optimal value is
  attained by some input (not necessarily by the value of :math:`x` itself
  when it sits on a threshold; see below).
- In ``"leaf"``, ``"ocean"`` and ``"biggs_perakis"`` each tree may take a
  different branch at the same threshold. The resulting combination of leaves
  may correspond to no input at all, and the optimal objective can be larger
  (for a maximization) than any value the ensemble can predict. In
  ``"leaf"`` and ``"biggs_perakis"``, each tree can additionally violate its
  own constraints within the feasibility tolerance, with the same effect.

A positive ``epsilon`` removes the ambiguity at thresholds. Its scope differs:

- In ``"leaf"``, ``"ocean"`` and ``"biggs_perakis"`` it only constrains the
  splits on the path taken by each tree.
- In ``"misic"`` and ``"misic_lazy"`` it applies to every threshold of
  the ensemble, since every :math:`z_{i,j}` is linked to :math:`x_i`: the
  interval :math:`(\theta_{i,j}, \theta_{i,j} + \epsilon)` is excluded for all
  thresholds.
  The model becomes infeasible if an input variable cannot avoid all those
  intervals, and an :math:`\epsilon` below Gurobi's
  :external+gurobi:ref:`IntFeasTol <parameterintfeastol>` may not be enforced,
  because the indicator constraints are handled as SOS constraints. A warning
  is emitted when ``epsilon`` is positive for these formulations.

Because :math:`x` often sits exactly on a threshold, recomputing the prediction
of the regression at the value of :math:`x` (as
``get_error()`` does) may route :math:`x` down the other branch. This is
also the case with ``"misic"`` and ``"misic_lazy"``. To know which leaf
each tree selected, read the leaf variables in the
:py:attr:`tree_leaves
<gurobi_ml.modeling.decision_tree.decision_tree_model.TreeLeavesAccessor.tree_leaves>`
attribute of the returned object rather than comparing :math:`x` with the
thresholds.

Lazy Split Constraints
----------------------

The ``"misic_lazy"`` formulation is the ``"misic"`` formulation with the
constraints linking the leaf variables to the split variables marked as lazy
constraints: their Gurobi :external+gurobi:ref:`Lazy <attrlazy>` attribute is
set to 3.

:cite:t:`Misic2020` observed that only a few of those constraints are needed to
solve many instances, and generated them on demand.

The effect depends on the instance. It is usually beneficial when few
branch-and-bound nodes are needed, since most split constraints then never
enter the model. It can be slower when the search branches a lot, because
after the root, lazy constraints are only checked against integer solutions
and the relaxation is weaker than with the constraints in the model. The set
of feasible solutions is the same as with ``"misic"``.

Choosing a Formulation
----------------------

No formulation is best on all problems. The following observations come from
experiments that maximize the prediction of the ensemble over a box of inputs,
and should be confirmed on problems of the intended shape:

- ``"misic"`` and ``"misic_lazy"`` are the only formulations whose
  optimal value is always attained by some input. ``"misic"`` was the fastest
  on large ensembles of deep trees, where its number of binary variables stays
  bounded by the number of distinct thresholds.
- ``"leaf"`` was the fastest when additional constraints restrict the
  solution to a small region of the input space.
- ``"biggs_perakis"`` was the fastest on random forests whose trees share few
  thresholds, and slower than ``"leaf"`` elsewhere. It has the shortest build
  time on large ensembles of deep trees, which matters when many models are
  built and each is solved quickly.
- ``"ocean"`` was the fastest when the features take very few distinct values,
  but it has the same exactness issue at thresholds as ``"leaf"``.
- The difficulty of the optimization problem depends more on the trained
  ensemble (framework, depth, regularization) than on the formulation.
