.. Gurobi Machine Learning documentation master file, created by
   sphinx-quickstart on Tue Jul 12 10:14:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Gurobi Machine Learning!
===================================

Gurobi Machine Learning is a Python package to help using *trained* regression models in a mathematical optimization
model. The package supports a variety of regression models (linear, logistic, neural networks, decision trees,...) trained by
different machine learning frameworks (scikit-learn, Keras and PyTorch).

When the regression model is embedded in an optimization model, its input and output are decision variables.
They are linked by the regression in that the output variables values are the values predicted by the regression from the input variables.

The aim of the package is to:

   #. Provide a simple way to embed regression models in a gurobipy model.
   #. Promote a deeper integration between predictive and prescriptive analytics.
   #. Allow to easily compare the effect of using different regression models in the same mathematical optimization application.
   #. Improve the algorithmic performance of Gurobi on those models.

We give below basic instalation and usage instructions.

Install
-------

The package should be installed with pip (or add it to your `requirements.txt` file):


.. code-block:: console

  (.venv) pip install gurobi-machinelearning


.. note::

  If not already installed this should install the gurobipy and
  numpy packages.

  The package needs Gurobi version 10 to work properly.

  The package has been tested with and is supported for python 3.9 and python 3.10


Usage
-----

The package essentially provides one function: :py:func:`gurobi_ml.add_predictor_constr`.
The function takes as arguments: a gurobipy model, a supported regression model,
input (Gurobi) variables for the regression model and output (Gurobi) variables
for the regression models.

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

For the list of models supported please refer to the :doc:`supported`, section.
Note that even though the :doc:`api/add_predictor_constr`, should add the correct model for any supported predictor, individual
functions for each predictor are also available.

For some regression models additional optional parameters can be set to tune the way the predictor is inserted in the Gurobi model. Those are documented
in the corresponding function linked from :doc:`supported`.

For a simple abstract example on how to use the package please refer to the :doc:`quickstart` section. More advanced examples are avaiable in the
:doc:examples section.

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
