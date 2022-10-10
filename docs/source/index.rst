.. ml2gurobi documentation master file, created by
   sphinx-quickstart on Tue Jul 12 10:14:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ml2gurobi's documentation!
=====================================

**ml2gurobi** is a Python package to insert machine learning models trained by
popular frameworks (scikit-learn, Keras, Pytorch) in a Gurobi optimization models.
The trained predictor can then be used in the optimization to model relationships between
decision variables.
The goal of the package is to provide a straightforward way
to Currently, the package only deals with regression models. The following
models are supported:

 - Linear regression
 - Logistic regression
 - Neural network regressor
 - Decision tree regressor
 - Gradient Boosting tree
 - Random Forest

All those regression models trained by scikit-learn are supported.


.. note::

   This project is under active development.


.. toctree::
   install
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
