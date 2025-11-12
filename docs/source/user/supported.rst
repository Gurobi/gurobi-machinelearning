Supported Regression models
###########################

The package currently support various `scikit-learn
<https://scikit-learn.org/stable/>`_ objects. It also support
Gradient Boosting Regreesion from `XGboost <https://xgboost.readthedocs.io/en/stable/>`_ and has limited support for
`Keras <https://keras.io/>`_ and `PyTorch <https://pytorch.org/>`_. Only
sequential neural networks with ReLU activation function are currently
supported. In :ref:`Mixed Integer Formulations`, we briefly outline the formulations used for the various
regression models.

The versions of those packages tested with the current version (|version|) are
listed in the table :ref:`table-versions`.


Scikit-learn
------------
The following tables list the name of the models supported, the name of the
corresponding object in the Python framework, and the function that can be used
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
     - :py:func:`add_linear_regression_constr
       <gurobi_ml.sklearn.linear_regression.add_linear_regression_constr>`
   * - Partial Least Square
     - :external:py:class:`PLSRegression
       <sklearn.cross_decomposition.PLSRegression>`
     - :py:func:`add_pls_regression_constr <gurobi_ml.sklearn.pls_regression.add_pls_regression_constr>`
   * - Logistic Regression [#]_
     - :external:py:class:`LogisticRegression
       <sklearn.linear_model.LogisticRegression>`
     - :py:func:`add_logistic_regression_constr
       <gurobi_ml.sklearn.logistic_regression.add_logistic_regression_constr>`
   * - Neural-network [#]_
     - :external:py:class:`MLPRegressor
       <sklearn.neural_network.MLPRegressor>`
     - :py:func:`add_mlp_regressor_constr
       <gurobi_ml.sklearn.mlpregressor.add_mlp_regressor_constr>`
   * - Decision tree
     - :external:py:class:`DecisionTreeRegressor
       <sklearn.tree.DecisionTreeRegressor>`
     - :py:func:`add_decision_tree_regressor_constr
       <gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor_constr>`
   * - Gradient boosting
     - :external:py:class:`GradientBoostingRegressor
       <sklearn.ensemble.GradientBoostingRegressor>`
     - :py:func:`add_gradient_boosting_regressor_constr
       <gurobi_ml.sklearn.gradient_boosting_regressor.add_gradient_boosting_regressor_constr>`
   * - Random Forest
     - :external:py:class:`RandomForestRegressor
       <sklearn.ensemble.RandomForestRegressor>`
     - :py:func:`add_random_forest_regressor_constr
       <gurobi_ml.sklearn.random_forest_regressor.add_random_forest_regressor_constr>`


.. list-table:: Transformers in :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Transformer
     - Scikit-learn object
     - Function to insert
   * - StandardScaler
     - :external:py:class:`StandardScaler
       <sklearn.preprocessing.StandardScaler>`
     - :py:func:`add_standard_scaler_constr
       <gurobi_ml.sklearn.preprocessing.add_standard_scaler_constr>`
   * - Pipeline
     - :external:py:class:`Pipeline <sklearn.pipeline.Pipeline>`
     - :py:func:`add_pipeline_constr <gurobi_ml.sklearn.pipeline.add_pipeline_constr>`
   * - PolynomialFeatures [#]_
     - :external:py:class:`PolynomialFeatures
       <sklearn.preprocessing.PolynomialFeatures>`
     - :py:func:`add_polynomial_features_constr
       <gurobi_ml.sklearn.preprocessing.add_polynomial_features_constr>`
   * - ColumnTransformer
     - :external:py:class:`ColumnTransformer
       <sklearn.compose.ColumnTransformer>`
     - :py:mod:`add_column_transformer_constr
       <gurobi_ml.sklearn.column_transformer.add_column_transformer_constr>`

Keras
-----

`Keras <https://keras.io/>`_ neural networks are generated either using the
`functional API <https://keras.io/guides/functional_api/>`_, `subclassing model
<https://keras.io/guides/making_new_layers_and_models_via_subclassing/>`_ or the
`Sequential <https://keras.io/api/models/sequential/>`_ class.

They can be formulated in a Gurobi model with the function
:py:func:`add_keras_constr <gurobi_ml.keras.add_keras_constr>`.

Currently, only two types of layers are supported:

    * `Dense layers <https://keras.io/api/layers/core_layers/dense/>`_ (possibly
      with `relu` activation),
    * `ReLU layers <https://keras.io/api/layers/activation_layers/relu/>`_ with
      default settings.

PyTorch
-------


In PyTorch, only :external+torch:py:class:`torch.nn.Sequential` objects are
supported.

They can be formulated in a Gurobi model with the function
:py:func:`add_sequential_constr <gurobi_ml.torch.sequential.add_sequential_constr>`.

Currently, only two types of layers are supported:

   * :external+torch:py:class:`Linear layers <torch.nn.Linear>`,
   * :external+torch:py:class:`ReLU layers <torch.nn.ReLU>`.

ONNX
----

`ONNX <https://onnx.ai/>`_ models for sequential multi-layer perceptrons are
supported when composed of `Gemm` (dense) operators and `Relu` activations.

They can be formulated in a Gurobi model with the function
:py:func:`add_onnx_constr <gurobi_ml.onnx.onnx_model.add_onnx_constr>`.

Currently, only the following are supported:

   * `Gemm` nodes with default attributes (`alpha=1`, `beta=1`) and optional
     `transB` attribute,
   * `Relu` activations.

XGBoost
-------

XGboost's :external+xgb:py:class:`xgboost.Booster` can be formulated in a Gurobi model
with the function :py:func:`add_xgboost_regressor_constr <gurobi_ml.xgboost.xgboost_regressor.add_xgboost_regressor_constr>`.
The scikit-learn wrapper :external+xgb:py:class:`xgboost.XGBRegressor` can be formulated
using :py:func:`add_xgbregressor_constr <gurobi_ml.xgboost.xgboost_regressor.add_xgbregressor_constr>`.

Currently only "gbtree" boosters are supported. Note that all options of "grbtree" may not be supported. In particular,
those that may result in a different prediction function. For the "objective" option, we only support the default `reg:squarederror`
and `reg:logistic`.
If you encounter an issue don't hesitate to contact us.

LightGBM
--------

LightGBM's :external+lightgbm:py:class:`lightgbm.Booster` can be formulated in a Gurobi model
with the function :py:func:`add_lgbm_booster_constr <gurobi_ml.lightgbm.lgbm_regressor.add_lgbm_booster_constr>`.
The scikit-learn wrapper :external+lightgbm:py:class:`lightgbm.sklearn.LGBMRegressor` can be formulated
using :py:func:`add_lgbmregressor_constr <gurobi_ml.lightgbm.lgbm_regressor.add_lgbmregressor_constr>`.

Note that all options of LightGBM may not be supported. In particular,
those that may result in a different prediction function.
If you encounter an issue don't hesitate to contact us.

.. rubric:: Footnotes

.. [#] Only binary classification. The logsitic function is approximated by a piece wise linear function.
.. [#] Only networks with `"relu"` activation for hidden layers and `"identity"`
    for the output layer.
.. [#] Only polynomial features of degree 2.
