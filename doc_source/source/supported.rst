Supported Regression models
===========================

The package currently support various `scikit-learn <https://scikit-learn.org/stable/>`_ objects and has limited support
for `Keras <https://keras.io/>`_ and `PyTorch <https://pytorch.org/>`_ neural networks with ReLU activation.

Scikit-learn
------------
The following tables list the name of the models supported,
the name of the corresponding object in the python framework,
and the function that can be used to insert it in a gurobi model

.. list-table:: Supported regression models of :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Regression Model
     - Scikit-learn object
     - Function to insert
   * - Ordinary Least Square
     - :external:py:class:`LinearRegression <sklearn.linear_model.LinearRegression>`
     - :doc:`api/add_linear_regression_constr`
   * - Logistic regression
     - :external:py:class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
     - :doc:`api/add_logistic_regression_constr`
   * - Neural-network
     - :external:py:class:`MLPRegressor <sklearn.neural_network.MLPRegressor>`
     - :doc:`api/add_mlp_regressor_constr`
   * - Decision tree
     - :external:py:class:`DecisionTreeRegressor <sklearn.tree.DecisionTreeRegressor>`
     - :doc:`api/add_decision_tree_regressor_constr`
   * - Gradient boosting
     - :external:py:class:`GradientBoostingRegressor <sklearn.ensemble.GradientBoostingRegressor>`
     - :doc:`api/add_gradient_boosting_regressor_constr`
   * - Random Forest
     - :external:py:class:`RandomForestRegressor <sklearn.ensemble.RandomForestRegressor>`
     - :doc:`api/add_random_forest_regressor_constr`


.. list-table:: Transformers in scikit-learn
   :widths: 25 25
   :header-rows: 1

   * - Scikit-learn object
     - Function to insert
   * - :external:py:class:`StandardScaler <sklearn.preprocessing.StandardScaler>`
     - :doc:`api/add_standard_scaler_constr`
   * - :external:py:class:`PolynomialFeatures <sklearn.preprocessing.PolynomialFeatures>`
     - :doc:`api/add_polynomial_features_constr`

Keras
-----

 * Keras
    * Linear layers
    * ReLU layers

PyTorch
-------

In PyTorch, only :external+torch:py:class:`torch.nn.Sequential` objects are supported.
Also, the only supported layers are:

   * :external+torch:py:class:`Linear layers <torch.nn.Linear>`, and
   * :external+torch:py:class:`ReLU layers <torch.nn.ReLU>`.

They can be embedded in a Gurobi model with the function :doc:`api/add_sequential_constr`.
