Supported Regression models
===========================

The package currently support various `scikit-learn <https://scikit-learn.org/stable/>`_ objects.
It also  has limited support
for `Keras <https://keras.io/>`_ and `PyTorch <https://pytorch.org/>`_.
Only sequential neural networks with ReLU activation function are currently supported.


.. list-table:: The following packages are supported in the current version (|version|)
   :widths: 50 50
   :header-rows: 1

   * - Package
     - Version
   * - ``pandas``
     - |PandasVersion|
   * - ``torch``
     - |TorchVersion|
   * - ``scikit-learn``
     - |SklearnVersion|
   * - ``tensorflow``
     - |TensorflowVersion|



Scikit-learn
------------
The following tables list the name of the models supported,
the name of the corresponding object in the python framework,
and the function that can be used to insert it in a Gurobi model.

.. list-table:: Supported regression models of :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Regression Model
     - Scikit-learn object
     - Function to insert
   * - Ordinary Least Square
     - :external:py:class:`LinearRegression <sklearn.linear_model.LinearRegression>`
     - :py:func:`add_linear_regression_constr <gurobi_ml.sklearn.add_linear_regression_constr>`
   * - Logistic regression
     - :external:py:class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
     - :py:func:`add_logistic_regression_constr <gurobi_ml.sklearn.add_logistic_regression_constr>`
   * - Neural-network [#]_
     - :external:py:class:`MLPRegressor <sklearn.neural_network.MLPRegressor>`
     - :py:func:`add_mlp_regressor_constr <gurobi_ml.sklearn.add_mlp_regressor_constr>`
   * - Decision tree
     - :external:py:class:`DecisionTreeRegressor <sklearn.tree.DecisionTreeRegressor>`
     - :py:func:`add_decision_tree_regressor_constr <gurobi_ml.sklearn.add_decision_tree_regressor_constr>`
   * - Gradient boosting
     - :external:py:class:`GradientBoostingRegressor <sklearn.ensemble.GradientBoostingRegressor>`
     - :py:func:`add_gradient_boosting_regressor_constr <gurobi_ml.sklearn.add_gradient_boosting_regressor_constr>`
   * - Random Forest
     - :external:py:class:`RandomForestRegressor <sklearn.ensemble.RandomForestRegressor>`
     - :py:func:`add_random_forest_regressor_constr <gurobi_ml.sklearn.add_random_forest_regressor_constr>`


.. list-table:: Transformers in :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Transformer
     - Scikit-learn object
     - Function to insert
   * - StandardScaler
     - :external:py:class:`StandardScaler <sklearn.preprocessing.StandardScaler>`
     - :py:func:`gurobi_ml.sklearn.add_standard_scaler_constr`
   * - Pipeline
     - :external:py:class:`Pipeline <sklearn.pipeline.Pipeline>`
     - :py:func:`gurobi_ml.sklearn.add_pipeline_constr`
   * - PolynomialFeatures
     - :external:py:class:`PolynomialFeatures <sklearn.preprocessing.PolynomialFeatures>` [#]_
     - :py:func:`gurobi_ml.sklearn.add_polynomial_features_constr`

Keras
-----

`Keras <https://keras.io/>`_ neural networks generated either using the `functional API <https://keras.io/guides/functional_api/>`_,
`subclassing model <https://keras.io/guides/making_new_layers_and_models_via_subclassing/>`_ or the
`Sequential <https://keras.io/api/models/sequential/>`_ class.

They can be embedded in a Gurobi model with the function :py:func:`gurobi_ml.keras.add_keras_constr`.

Currently, only two types of layers are supported:

    * `Dense layers <https://keras.io/api/layers/core_layers/dense/>`_ (possibly with `relu` activation),
    * `ReLU layers <https://keras.io/api/layers/activation_layers/relu/>`_ with default settings.

PyTorch
-------


In PyTorch, only :external+torch:py:class:`torch.nn.Sequential` objects are supported.

They can be embedded in a Gurobi model with the function :py:func:`gurobi_ml.torch.add_sequential_constr`.

Currently, only two types of layers are supported:

   * :external+torch:py:class:`Linear layers <torch.nn.Linear>`,
   * :external+torch:py:class:`ReLU layers <torch.nn.ReLU>`.

.. rubric:: Footnotes

.. [#] Only networks with `"relu"` activation for hidden layers and `'identity'` for the output layer.
.. [#] Only polynomial features of degree 2.
