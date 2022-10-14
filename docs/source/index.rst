.. Gurobi Machine Learning documentation master file, created by
   sphinx-quickstart on Tue Jul 12 10:14:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Gurobi Machine Learning's documentation!
===================================================

Gurobi Machine Learning is a Python package to insert machine learning regression models trained by
popular frameworks in a gurobipy optimization models.

That way, the trained regression model can be used in the optimization to establish relationships between
decision variables.

The goal of the package is to:
  #. Simplify the process of importing a trained machine learning model built with a popular ML package into an optimization model.
  #. Improve algorithmic performance to enable the optimization model to explore a sizable space of solutions that satisfy the variable relationships captured in the ML model.
  #. Make it easier for optimization models to mix explicit and implicit constraints.

Supported Regression models
===========================
The package currently support various object from scikit-learn and has limited support
for Keras and PyTorch neural networks with ReLU activation.

Scikit-learn
------------
The following tables list the name of the models supported,
the name of the corresponding object in the python framework,
and the function that can be used to insert it in a gurobi model


.. list-table:: Regression models in scikit-learn
   :widths: 25 25 50
   :header-rows: 1

   * - Regression Model
     - Scikit-learn object
     - Function to insert
   * - Linear regression
     - :external:py:class:`LinearRegression <sklearn.linear_model.LinearRegression>`
     - add_linear_regression
   * - Logistic regression
     - :external:py:class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
     - add_logistic_regression
   * - Neural-network
     - :external:py:class:`MLPRegressor <sklearn.neural_network.MLPRegressor>`
     - add_mlp_regressor
   * - Decision tree
     - :external:py:class:`DecisionTreeRegressor <sklearn.tree.DecisionTreeRegressor>`
     - add_decision_tree_regressor
   * - Gradient boosting
     - :external:py:class:`GradientBoostingRegressor <sklearn.ensemble.GradientBoostingRegressor>`
     - add_gradient_boosting_regressor
   * - Random Forest
     - :external:py:class:`RandomForestRegressor <sklearn.ensemble.RandomForestRegressor>`
     - add_random_forest_regressor


.. list-table:: Transformers in scikit-learn
   :widths: 25 25
   :header-rows: 1

   * - Scikit-learn object
     - Function to insert
   * - :external:py:class:`StandardScaler <sklearn.preprocessing.StandardScaler>`
     - add_standard_scaler
   * - :external:py:class:`PolynomialFeatures <sklearn.preprocessing.PolynomialFeatures>`
     - add_polynomial_features

Keras
-----

 * Keras
    * Dense layers
    * ReLU layers

PyTorch
-------
 * PyTorch (only torch.nn.Sequential objects)
    * Dense layers
    * ReLU layers



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
