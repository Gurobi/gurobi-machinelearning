import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from joblib import Parallel, delayed, dump, load, parallel_backend
from sklearn.ensemble import GradientBoostingRegressor  # noqa
from sklearn.ensemble import RandomForestRegressor  # noqa
from sklearn.linear_model import LinearRegression  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.neural_network import MLPRegressor  # noqa
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor  # noqa

from gurobi.machinelearning import add_predictor_constr
from gurobi.machinelearning.sklearn import sklearn_predictors

KNOWN_FEATURES = ["SAT", "GPA"]
DEC_FEATURES = ["scholarship"]


def do_regression(regressor, known_features, dec_features, **kwargs):
    # Retrieve historical data used to do the regression
    historical_data = pd.read_csv("data/college_student_enroll-s1-1.csv", index_col=0)

    # Classify our features between the ones that are fixed and the ones that will be
    # part of the optimization problem
    features = known_features + dec_features

    # The target for training
    target = "enroll"

    historical_data = historical_data[features + [target]]

    # Run our regression
    X = historical_data.loc[:, features]
    Y = historical_data.loc[:, "enroll"]
    scaler = StandardScaler()

    pipe = make_pipeline(scaler, regressor)
    pipe.fit(X=X, y=Y)

    decidx = historical_data.columns.get_indexer(dec_features)
    return (pipe, decidx)


def do_model(pipe, decidx, known_features, dec_features, reluformulation=None):
    # ### Do the optimization model
    # Retrieve new data used for the optimization
    studentsdata = pd.read_csv("data/admissions500.csv", index_col=0)
    studentsdata = studentsdata[known_features]
    # Check that features are in identical order

    features = known_features + dec_features

    nstudents = studentsdata.shape[0]

    # Start with classical part of the model
    m = gp.Model()

    knownidx = studentsdata.columns.get_indexer(known_features)
    assert max(knownidx) + 1 == len(knownidx)
    decidx = np.arange(len(dec_features)) + len(knownidx)

    lb = np.zeros((nstudents, len(features)))
    ub = np.ones((nstudents, len(features))) * 2.5
    lb[:, knownidx] = studentsdata.loc[:, known_features]
    ub[:, knownidx] = studentsdata.loc[:, known_features]
    x = m.addMVar((nstudents, len(features)), lb=lb, ub=ub, name="x")
    y = m.addMVar((nstudents, 1), lb=-GRB.INFINITY, name="y")

    m.setObjective(y[:, 0].sum(), gp.GRB.MAXIMIZE)
    m.addConstr(x[:, decidx].sum() <= 0.2 * nstudents)

    # create transforms to turn scikit-learn pipeline into Gurobi constraints
    pipe2gurobi = add_predictor_constr(m, pipe, x, y)
    return m


def dojanosformulation(pipe):
    (pipe, decidx) = pipe
    model = do_model(pipe, decidx, KNOWN_FEATURES, DEC_FEATURES)
    return model


def paramstring(params):
    rval = {}
    for name, value in params.items():
        if isinstance(value, list):
            value = "x".join([f"{v}" for v in value])
        rval[name] = value
    if len(rval):
        return "_{}".format("-".join([f"{n}={v}" for n, v in rval.items()]))
    else:
        return ""


def gen_nn(totest):
    regressor = totest["regressor"]
    kwargs = totest["kwargs"]
    print(regressor, kwargs)
    filename = "{}_{}{}.joblib".format("Janos", regressor.lower(), paramstring(kwargs))
    try:
        load(filename)
        return
    except Exception:
        pass
    regressor = eval(regressor)(**kwargs)
    pipe, decidx = do_regression(regressor, KNOWN_FEATURES, DEC_FEATURES)
    nn = pipe.steps[-1][1]
    dump((pipe, decidx), filename)


def regressor_params(name):
    if name == "MLPRegressor":
        return [
            {"hidden_layer_sizes": [5] * 2},
            {"hidden_layer_sizes": [5] * 3},
            {"hidden_layer_sizes": [10] * 2},
            {"hidden_layer_sizes": [10] * 3},
            {"hidden_layer_sizes": [15] * 2},
            {"hidden_layer_sizes": [15] * 3},
        ]
    if name == "GradientBoostingRegressor":
        rval = []
        for n_estimators in [5, 10, 15, 20]:
            for depth in [4, 5]:
                for max_leaf in [5, 7]:
                    rval.append({"n_estimators": n_estimators, "max_depth": depth, "max_leaf_nodes": max_leaf})
        return rval
    if name == "RandomForestRegressor":
        rval = []
        for n_estimators in [5, 10, 15, 20]:
            for depth in [4, 5]:
                for max_leaf in [5, 7]:
                    rval.append({"n_estimators": n_estimators, "max_depth": depth, "max_leaf_nodes": max_leaf})
        return rval
    if name == "DecisionTreeRegressor":
        rval = []
        for depth in [5, 10, 15]:
            for max_leaf in [5, 10, 20, 30, 40]:
                rval.append({"max_depth": depth, "max_leaf_nodes": max_leaf})
        return rval
    if name == "LogisticRegression":
        rval = []
        for penalty in ["l1", "l2", "elasticnet"]:
            for c in [1e-2, 1e-1, 1e0, 1e1]:
                rval.append({"solver": "saga", "C": c, "penalty": penalty, "l1_ratio": 0.5})
        return rval
    return [{}]


def gen_all_nn():
    do = gen_nn
    excluded = ["LinearRegression", "MLPClassifier"]
    regressors = [r for r in sklearn_predictors().keys() if r not in excluded]
    to_test = []
    noseed = ["GradientBoostingRegressor", "RandomForestRegressor", "DecisionTreeRegressor", "LogisticRegression"]
    for reg in regressors:
        params = regressor_params(reg)
        if reg in noseed:
            for par in params:
                to_test.append({"regressor": reg, "kwargs": par})
        else:
            for par in params:
                for seed in range(10):
                    par["random_state"] = seed
                    to_test.append({"regressor": reg, "kwargs": par})
    with parallel_backend("multiprocessing"):
        Parallel(n_jobs=4, verbose=10)(delayed(do)(test) for test in to_test)


if __name__ == "__main__":
    gen_all_nn()
