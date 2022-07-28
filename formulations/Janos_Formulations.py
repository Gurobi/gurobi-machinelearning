import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from joblib import Parallel, delayed, dump, load, parallel_backend
from ml2gurobi.sklearn import Pipe2Gurobi
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KNOWN_FEATURES = ["SAT", "GPA"]
DEC_FEATURES = ["scholarship"]


def do_regression(layers, seed, known_features, dec_features):
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

    regression = MLPRegressor(hidden_layer_sizes=layers, random_state=seed)
    pipe = make_pipeline(scaler, regression)
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
    pipe2gurobi = Pipe2Gurobi(pipe, m)
    if reluformulation is not None:
        pipe2gurobi.steps[-1].actdict["relu"] = reluformulation

    # Add constraint to predict value of y using kwnown and to compute features
    pipe2gurobi.predict(X=x, y=y)

    m._pipe2gurobi = pipe2gurobi
    return m


def dojanosformulation(pipe):
    (pipe, decidx) = pipe
    model = do_model(pipe, decidx, KNOWN_FEATURES, DEC_FEATURES)
    return model


def gen_nn(layers, seed):
    filename = "{}_nn-{}_seed{}.joblib".format("Janos", "-".join([f"{n}" for n in layers]), seed)
    try:
        load(filename)
        return
    except Exception:
        pass
    pipe, decidx = do_regression(layers, seed, KNOWN_FEATURES, DEC_FEATURES)
    nn = pipe.steps[-1][1]
    for layer in nn.coefs_:
        layer[np.abs(layer) < 1e-8] = 0.0
    dump((pipe, decidx), filename)


def gen_all_nn():
    do = gen_nn
    with parallel_backend("multiprocessing"):
        Parallel(n_jobs=4, verbose=10)(
            delayed(do)(hidden_layers, seed)
            for hidden_layers in (
                [5] * 2,
                [5] * 3,
                [10] * 2,
                [10] * 3,
                [15] * 2,
                [15] * 3,
            )
            for seed in range(10)
        )


if __name__ == "__main__":
    gen_all_nn()
