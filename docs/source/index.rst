.. Gurobi Machine Learning documentation master file, created by
   sphinx-quickstart on Tue Jul 12 10:14:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Gurobi Machine Learning!
===================================

Gurobi Machine Learning is a Python package to embed *trained* regression models in a mathematical optimization
model. The package supports a variety of regression models (linear, logistic, neural networks, decision trees,...) trained by
different machine learning framework (scikit-learn, Keras and PyTorch).
When the regression model is embedded in an optimization model, its input and output are decision variables.
They are linked by the regression in that the output variables values are the values predicted by the regression from the input variables.

The aim of the package is to:

   #. Provide a simple way to embed regression models in a gurobipy model.
   #. Promote a deeper integration between predictive and prescriptive analytics.
   #. Allow to easily compare the effect of using different regression models in the same mathematical optimization application.
   #. Improve the algorithmic performance of Gurobi on those models.

Contents
--------
.. toctree::
   install
   example
   supported
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
