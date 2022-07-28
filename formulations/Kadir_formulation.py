"""
Created on Mon Jul 19 17:26:18 2021

@author: 4ka
"""
import os

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from joblib import Parallel, delayed, dump, load, parallel_backend
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

# import my functions
from gurobi.machinelearning.sklearn import Pipe2Gurobi


def do_regression(layers, seed):
    X = np.genfromtxt("X.csv")
    Y = np.genfromtxt("Y.csv")

    regression = MLPRegressor(
        hidden_layer_sizes=layers,
        verbose=True,
        max_iter=500,
        activation="relu",
        early_stopping=True,
        random_state=seed,
    )
    pipe = make_pipeline(regression)
    pipe.fit(X, Y)
    return pipe


def do_model(pipe, seed=None, reluformulation=None):
    if seed is not None:
        np.random.seed = seed
    else:
        np.random.seed = pipe.steps[-1][1].random_state
    p = np.random.randint(30, size=24)
    m = gp.Model()
    x = m.addMVar((1, 48), vtype=GRB.CONTINUOUS, name="x", lb=0, ub=410)
    y = m.addMVar((1, 24), lb=20, ub=30, vtype=GRB.CONTINUOUS, name="y")
    m.setObjective(p @ x[0, 24:], GRB.MINIMIZE)

    pipe2gurobi = Pipe2Gurobi(pipe, m)
    if reluformulation is not None:
        pipe2gurobi.steps[-1].actdict["relu"] = reluformulation
    pipe2gurobi.predict(x, y)

    m._pipe2gurobi = pipe2gurobi
    return m


def heuristic(nn, p, nn2gurobi):
    X = np.genfromtxt("X.csv")

    def prop():
        pass

    prediction = nn.forward(X)
    feasibles = X[((prediction >= 20) & (prediction <= 30)).all(axis=1), :]
    sortedinputs = np.argsort(p @ feasibles[:, 24:].numpy().T)

    prop(nn2gurobi, feasibles[sortedinputs[0]].numpy().reshape(1, -1), reset=True)


def gen_nn(layers, seed):
    filename = "{}_nn-{}_seed{}.joblib".format("Kadir", "-".join([f"{n}" for n in layers]), seed)
    try:
        load(filename)
        return
    except Exception:
        pass
    pipe = do_regression(layers, seed)
    nn = pipe.steps[-1][1]
    for layer in nn.coefs_:
        layer[np.abs(layer) < 1e-8] = 0.0
    dump(pipe, filename)


def doone(filename, doobbt=None, seed=None):
    outputfile = filename.strip(".joblib") + f"-objseed{seed}.lp.bz2"
    try:
        gp.read(outputfile)
        return
    except Exception:
        pass

    pipe = load(f"Networks/{filename}")
    if pipe.steps[-1][1].hidden_layer_sizes != [128, 128, 128]:
        return

    if filename.startswith("Kadir"):
        m = do_model(pipe, seed)
    else:
        return
    if doobbt:
        m._pipe2gurobi.steps[-1].obbt()

    m.write(outputfile)


def gen_all_nn():
    with parallel_backend("multiprocessing"):
        do = gen_nn
        Parallel(n_jobs=8, verbose=10)(
            delayed(do)(hidden_layers, seed)
            for hidden_layers in ([128] * 2, [128] * 3, [256] * 2, [256] * 3)
            for seed in range(10)
        )


if __name__ == "__main__":
    files = [f for f in os.listdir("Networks") if f.startswith("Kadir")]
    doobbt = False

    Parallel(n_jobs=5, verbose=10)(delayed(doone)(f, doobbt, seed) for f in files for seed in range(1, 11))
