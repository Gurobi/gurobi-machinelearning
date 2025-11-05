Introduction
############

The integration of Machine Learning (ML) techniques and Mathematical
Optimization is a topic of growing interest. One particular approach is to
be able to use *trained* ML models in optimization models
(:cite:t:`JANOS`, :cite:t:`Maragano.et.al2021`, :cite:t:`ceccon2022omlt`). In this approach, the
input and the prediction of the ML model are decision variables of the
optimization while its parameters are fixed. We say that the ML model is
formulated in the optimization model, and we refer to the input and predictions
as input and output variables respectively. They are linked in that, in a
feasible solution, the output variables values are the values predicted by the
ML model from the input variables values.

Gurobi Machine Learning is an :doc:`open-source <../meta/license>` Python package to formulate *trained
regression* models [#]_ in a :external+gurobi:py:class:`Model` to be
solved with the `Gurobi <https://www.gurobi.com>`_ solver.

The aim of the package is to:

   #. Provide a simple way to formulate regression models in a gurobipy model.
   #. Promote a deeper integration between predictive and prescriptive
      analytics.
   #. Allow to easily compare the effect of using different regression models in
      the same mathematical optimization application.
   #. Improve the algorithmic performance of Gurobi on those models.

The package currently supports various `scikit-learn
<https://scikit-learn.org/stable/>`_ objects. It can also formulate
gradient boosting regression models from `XGboost <https://xgboost.readthedocs.io/en/stable/>`_
and `LightGBM <https://lightgbm.readthedocs.io/en/stable/>`.
Finally, it has limited support for
`Keras <https://keras.io/>`_. Only neural networks with ReLU activation
can be used with these two packages.

The package is actively developed and users are encouraged to :doc:`contact us
<../meta/contactus>` if they have applications where they use a regression model
that is currently not available.

Below, we give basic installation and usage instructions.

Install
*******

We encourage to install the package via pip (or add it to your
`requirements.txt` file):


.. code-block:: console

  (.venv) pip install gurobi-machinelearning


.. note::

  If not already installed, this should install the :pypi:`gurobipy`, :pypi:`numpy` and :pypi:`scipy`
  packages.


.. note::

  The package is tested with and is supported for Python 3.9, 3.10, 3.11 and 3.12.
  It is also tested and supported with Gurobi 10, 11 and 12. Note however, that some newer
  features of Gurobi from later versions are used and some models may perform significantly
  worse with the older versions.

  The following table lists the version of the relevant packages that are
  tested and supported in the current version (|version|).

  .. _table-versions:

  .. list-table:: Supported packages with version |version|
     :widths: 50 50
     :align: center
     :header-rows: 1

     * - Package
       - Version
     * - :pypi:`numpy`
       - |NumpyVersion|
     * - :pypi:`scipy`
       - |ScipyVersion|
     * - :pypi:`pandas`
       - |PandasVersion|
     * - :pypi:`torch`
       - |TorchVersion|
     * - :pypi:`scikit-learn`
       - |SklearnVersion|
     * - :pypi:`keras`
       - |KerasVersion|
     * - :pypi:`xgboost`
       - |XGBoostVersion|
     * - :pypi:`lightgbm`
       - |LightGBMVersion|

  Installing any of the machine learning packages is only required if the
  predictor you want to insert uses them (i.e. to insert a Keras based predictor
  you need to have :pypi:`keras` installed).


Usage
*****

The main function provided by the package is
:py:func:`gurobi_ml.add_predictor_constr`. It takes as arguments: a :external+gurobi:py:class:`Model`, a
:doc:`supported regression model <supported>`, input `Gurobi variables
<https://www.gurobi.com/documentation/current/refman/variables.html>`_ and
output `Gurobi variables
<https://www.gurobi.com/documentation/current/refman/variables.html>`_.

By invoking the function, the :external+gurobi:py:class:`Model` is augmented with variables and
constraints so that, in a solution, the values of the output variables are
predicted by the regression model from the values of the input variables. More
formally, if we denote by :math:`g` the prediction function of the regression
model, by :math:`x` the input variables and by :math:`y` the output variables,
then :math:`y = g(x)` in any solution.

The function :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>`
returns a modeling object derived from the class
:py:class:`AbstractPredictorConstr
<gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr>`. That object keeps track of all
the variables and constraints that have been added to the :external+gurobi:py:class:`Model` to
establish the relationship between input and output variables of the regression.

The modeling object can perform a few tasks:

   * Everything it created (i.e. variables and constraints to establish the
     relationship between input and output) can be removed with the
     :py:meth:`remove <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.remove>`
     method.
   * It can print a summary of what it added with the :py:meth:`print_stats
     <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.print_stats>` method.
   * Once Gurobi computed a solution to the optimization problem, it can compute
     the difference between what the regression model predicts from the input
     values and the values of the output variables in Gurobi's solution with the
     :py:meth:`get_error
     <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.print_stats>` method.


The function :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` is
a shorthand that should add the correct model for any supported regression
model, but individual functions for each regression model are also available.
For the list of frameworks and regression models supported, and the corresponding
functions please refer to the :doc:`supported` section. We also briefly
outline how the various regression models are expressed in Gurobi in the :ref:`Mixed Integer Formulations`
section.

For some regression models, additional optional parameters can be set to tune
the way the predictor is inserted in the Gurobi model. Those are documented in
the corresponding function linked from :doc:`supported`.

For a simple example on how to use the package please refer to
:doc:`../auto_userguide/example_simple`. More advanced examples are available
in the :doc:`../auto_examples/index` section.


.. note::

  Variable shapes: For tabular models (scikit-learn, tree ensembles, dense
  neural nets), inputs are typically 2D MVars with shape ``(batch, features)``
  and outputs are 1D or 2D (the package orients a 1D output based on the
  batch size). For convolutional neural networks (Keras/PyTorch), use 4D MVars
  with shape ``(batch, H, W, C)`` (channels-last).


.. rubric:: Footnotes

.. [#] Classification models are currently not supported (except binary logistic
    regression) but it is planned to add support to some models over time.
