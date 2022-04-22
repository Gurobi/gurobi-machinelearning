# ml-constraints
A simple framework to add machine learning based functions to a Gurobi model.

Namely allow to insert a function of the form y = g(x), where x and y are decision variables
(respectively input and output variaboes of g) and g is a predictor obtained from a trained
classifier/regreesor.

The package currently support for g:
- A standard scaler from scikit-learn
- A linear regression from scikit-learn
- A logistic regression from sicikit-learn
- A neural-network regression from scikit-learn (MLPRegressor)
- A sequential neural network from pytorch
