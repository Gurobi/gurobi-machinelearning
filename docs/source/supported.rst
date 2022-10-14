Supported Regression models
===========================
The package currently support various object `scikit-learn <https://scikit-learn.org/stable/> ` and has limited support
for `Keras <https://keras.io/>` and `PyTorch <https://pytorch.org/>` neural networks with ReLU activation.

Scikit-learn
------------
The following tables list the name of the models supported,
the name of the corresponding object in the python framework,
and the function that can be used to insert it in a gurobi model

:external+sklearn:std::`about`

.. list-table:: Supported regression models of :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Regression Model
     - Scikit-learn object
     - Function to insert
   * - Ordinary Least Square
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
