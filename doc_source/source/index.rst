.. Gurobi Machine Learning documentation master file, created by
   sphinx-quickstart on Tue Jul 12 10:14:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Gurobi Machine Learning!
===================================

Gurobi Machine Learning is a Python package to help using *trained* regression models in
mathematical optimization models. The package supports a variety of regression models
(linear, logistic, neural networks, decision trees,...) trained by
different machine learning frameworks (scikit-learn, Keras and PyTorch). They are inserted in a
`gurobipy model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_
to be solved with the `Gurobi <https://www.gurobi.com>`_ solver.

When the regression model is embedded in an optimization model, its input and output are decision variables.
They are linked by the regression in that the output variables values are the values predicted by the regression from the input variables in a feasible solution.

The aim of the package is to:

   #. Provide a simple way to embed regression models in a gurobipy model.
   #. Promote a deeper integration between predictive and prescriptive analytics.
   #. Allow to easily compare the effect of using different regression models in the same mathematical optimization application.
   #. Improve the algorithmic performance of Gurobi on those models.

The package currently support various `scikit-learn <https://scikit-learn.org/stable/>`_ objects and has limited support
for `Keras <https://keras.io/>`_ and `PyTorch <https://pytorch.org/>`_ neural networks with ReLU activation.
It is actively developed and users are encouraged to contact `Gurobi Optimization <https://www.gurobi.com>`_ if they have applications where they use a regression model that is currently
not available.

Below, we give basic installation and usage instructions.

Install
-------

We encourage to install the package via pip (or add it to your `requirements.txt` file):


.. code-block:: console

  (.venv) pip install gurobi-machinelearning


.. note::

  If not already installed, this should install the ``gurobipy`` and
  ``numpy`` packages.

  The package requires ``gurobipy`` version 10.0 or greater.

  The package has been tested with and is supported for Python 3.9 and Python 3.10.


Usage
-----

The package essentially provides one function: :py:func:`gurobi_ml.add_predictor_constr`.
The function takes as arguments: a `gurobipy model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_,
a supported regression model,
input (Gurobi) variables for the regression model and output (Gurobi) variables
for the regression model.

By calling the function, the gurobipy model is augmented with variables and
constraints so that, in a solution, the values of the output variables
are predicted by the regression model from the input variables.
More formally, if we denote by :math:`g` the prediction function of the regression model, by
:math:`x` the input variables
and by :math:`y` the output variables, in any solution we should have :math:`y = g(x)`.

The function :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` returns a so-called modeling object derived from the class
:py:class:`AbstractPredictorConstr <gurobi_ml.modeling.AbstractPredictorConstr>`.
That object keeps track of all the variables and constraints that have been added
to the gurobipy model to establish the relationship between input and output variables
of the regression.

The modeling object can perform a few tasks:

   * Everything it created (i.e. variables and constraints to establish the relationship between input and output)
     can be removed with the :py:meth:`remove <gurobi_ml.modeling.AbstractPredictorConstr.remove>` function.
   * It can print a summary of what it added with the :py:meth:`print_stats <gurobi_ml.modeling.AbstractPredictorConstr.print_stats>` function.
   * Once Gurobi computed a solution to the optimization problem, it can compute the difference between what the regression
     model predicts from the input values and the actual values of the output in Gurobi's solution with the
     :py:meth:`get_error <gurobi_ml.modeling.AbstractPredictorConstr.print_stats>` function.


For the list of frameworks and regression models supported please refer to the :doc:`supported` section.
Note that the function :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>`
should add the correct model for any supported predictor. Individual functions for each predictor are also available.

For some regression models, additional optional parameters can be set to tune the way the predictor is inserted in the Gurobi model. Those are documented
in the corresponding function linked from :doc:`supported`.

For a simple abstract example on how to use the package please refer to the :doc:`quickstart` section. More advanced examples are available in the
:doc:`examples` section.


Contents
--------
.. toctree::
   :maxdepth: 2

   quickstart
   supported
   examples
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
