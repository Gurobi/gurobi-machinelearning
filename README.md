# gurobi.machinelearning
A framework to add trained machine learning models as constraints to a gurobipy model.

Namely allow to insert a function of the form *y = g(x)*, where *x* and *y* are decision variables
(respectively input and output variables of *g*) and *g* is a predictor obtained from a trained
classifier/regressor.

The package currently support for g:
- A standard scaler from scikit-learn
- A linear regression from scikit-learn
- A logistic regression from sicikit-learn
- A neural-network regression from scikit-learn (MLPRegressor)
- A decision tree from scikit-learn
- A gradient boosting tree from scikit-learn
- A sequential neural network from pytorch
