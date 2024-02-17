.. only:: html

   .. image:: _static/gurobi_light.png
        :width: 220
        :target: https://www.gurobi.com
        :alt: Gurobi
        :align: right
        :class: float-right, only-light

   .. image:: _static/gurobi_dark.png
        :width: 220
        :target: https://www.gurobi.com
        :alt: Gurobi
        :align: right
        :class: float-right, only-dark


API
###

..  rst-class::  clear-both

Main Function
*************

The main function provided by the package formulates a predictor in
a Gurobi model. In most cases, it is the only function needed from
the package.

Some formulations of the predictors can have additional options
those are documented in the specific functions for each regression model.

.. currentmodule:: gurobi_ml

.. autosummary::
   :toctree: auto_generated/
   :caption: Main API

   add_predictor_constr


Scikit-learn API
****************

.. currentmodule:: gurobi_ml.sklearn

.. autosummary::
   :toctree: auto_generated/
   :caption: Scikit-learn API
   :template: modeling_object.rst

   column_transformer
   decision_tree_regressor
   gradient_boosting_regressor
   linear_regression
   logistic_regression
   mlpregressor
   pipeline
   pls_regression
   random_forest_regressor
   preprocessing


Keras API
*********

.. currentmodule:: gurobi_ml.keras

.. autosummary::
   :toctree: auto_generated/
   :caption: Keras API
   :template: modeling_object.rst

   keras

PyTorch API
***********

.. currentmodule:: gurobi_ml.torch

.. autosummary::
   :toctree: auto_generated/
   :caption: Pytorch API
   :template: modeling_object.rst

   sequential

XGBoost API
***********

.. currentmodule:: gurobi_ml.xgboost

.. autosummary::
   :toctree: auto_generated/
   :caption: XGBoost API
   :template: modeling_object.rst

   xgboost_regressor

LightGBM API
***********

.. currentmodule:: gurobi_ml.lightgbm

.. autosummary::
   :toctree: auto_generated/
   :caption: LightGBM API
   :template: modeling_object.rst

   lgbm_regressor

-------

.. toctree::
   :caption: Internal APIs
   :maxdepth: 1

   internal_apis/index
