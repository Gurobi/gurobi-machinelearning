Supported Regression models
===========================

The package currently support various `scikit-learn
<https://scikit-learn.org/stable/>`_ objects. It also has limited support for
`Keras <https://keras.io/>`_ and `PyTorch <https://pytorch.org/>`_. Only
sequential neural networks with ReLU activation function are currently
supported.

The versions of those packages tested with the current version (|version|) are
listed in the table :ref:`table-versions`.


Scikit-learn
------------
The following tables list the name of the models supported, the name of the
corresponding object in the python framework, and the function that can be used
to insert it in a Gurobi model.

.. list-table:: Supported regression models of :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Regression Model
     - Scikit-learn object
     - Function to insert
   * - Ordinary Least Square
     - :external:py:class:`LinearRegression
       <sklearn.linear_model.LinearRegression>`
     - :py:mod:`add_linear_regression_constr
       <gurobi_ml.sklearn.linear_regression>`
   * - Logistic regression [#]_
     - :external:py:class:`LogisticRegression
       <sklearn.linear_model.LogisticRegression>`
     - :py:mod:`add_logistic_regression_constr
       <gurobi_ml.sklearn.logistic_regression>`
   * - Neural-network [#]_
     - :external:py:class:`MLPRegressor
       <sklearn.neural_network.MLPRegressor>`
     - :py:mod:`add_mlp_regressor_constr
       <gurobi_ml.sklearn.mlpregressor>`
   * - Decision tree
     - :external:py:class:`DecisionTreeRegressor
       <sklearn.tree.DecisionTreeRegressor>`
     - :py:mod:`add_decision_tree_regressor_constr
       <gurobi_ml.sklearn.decision_tree_regressor>`
   * - Gradient boosting
     - :external:py:class:`GradientBoostingRegressor
       <sklearn.ensemble.GradientBoostingRegressor>`
     - :py:mod:`add_gradient_boosting_regressor_constr
       <gurobi_ml.sklearn.gradient_boosting_regressor>`
   * - Random Forest
     - :external:py:class:`RandomForestRegressor
       <sklearn.ensemble.RandomForestRegressor>`
     - :py:mod:`add_random_forest_regressor_constr
       <gurobi_ml.sklearn.random_forest_regressor>`


.. list-table:: Transformers in :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Transformer
     - Scikit-learn object
     - Function to insert
   * - StandardScaler
     - :external:py:class:`StandardScaler
       <sklearn.preprocessing.StandardScaler>`
     - :py:mod:`add_standard_scaler_constr
       <gurobi_ml.sklearn.add_standard_scaler_constr>`
   * - Pipeline
     - :external:py:class:`Pipeline <sklearn.pipeline.Pipeline>`
     - :py:mod:`add_pipeline_constr <gurobi_ml.sklearn.pipeline>`
   * - PolynomialFeatures
     - :external:py:class:`PolynomialFeatures [#]_
       <sklearn.preprocessing.PolynomialFeatures>`
     - :py:mod:`add_polynomial_features_constr
       <gurobi_ml.sklearn.add_polynomial_features_constr>`

Keras
-----

`Keras <https://keras.io/>`_ neural networks generated either using the
`functional API <https://keras.io/guides/functional_api/>`_, `subclassing model
<https://keras.io/guides/making_new_layers_and_models_via_subclassing/>`_ or the
`Sequential <https://keras.io/api/models/sequential/>`_ class.

They can be embedded in a Gurobi model with the function
:py:func:`gurobi_ml.keras.add_keras_constr`.

Currently, only two types of layers are supported:

    * `Dense layers <https://keras.io/api/layers/core_layers/dense/>`_ (possibly
      with `relu` activation),
    * `ReLU layers <https://keras.io/api/layers/activation_layers/relu/>`_ with
      default settings.

PyTorch
-------


In PyTorch, only :external+torch:py:class:`torch.nn.Sequential` objects are
supported.

They can be embedded in a Gurobi model with the function
:py:func:`gurobi_ml.torch.add_sequential_constr`.

Currently, only two types of layers are supported:

   * :external+torch:py:class:`Linear layers <torch.nn.Linear>`,
   * :external+torch:py:class:`ReLU layers <torch.nn.ReLU>`.

.. rubric:: Footnotes

.. [#] Only binary classification
.. [#] Only networks with `"relu"` activation for hidden layers and `'identity'`
    for the output layer.
.. [#] Only polynomial features of degree 2.
