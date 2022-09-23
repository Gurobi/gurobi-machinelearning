"""
Created on Mon Jul 19 17:26:18 2021

@author: 4ka
"""
import os

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from joblib import Parallel, delayed, parallel_backend

# import my functions
from gurobi.machinelearning.pytorch import Sequential


def do_regression(seed):
    X = torch.from_numpy(np.genfromtxt("X.csv")).float()
    Y = torch.from_numpy(np.genfromtxt("Y.csv")).float()

    hs = 128
    torch.manual_seed(seed)
    # Define a simple sequential network
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hs),
        torch.nn.ReLU(),
        torch.nn.Linear(hs, hs),
        torch.nn.ReLU(),
        torch.nn.Linear(hs, hs),
        torch.nn.ReLU(),
        torch.nn.Linear(hs, Y.shape[1]),
    )

    # Construct our loss function and an Optimizer.
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(1000):
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        pred = model(X)
        # Compute and print loss
        loss = criterion(pred, Y)
        if t % 100 == 0:
            print(f"iteration {t} loss: {loss.item()}")
        if loss.item() < 1e-4:
            break
        loss.backward()
        optimizer.step()
    else:
        print(f"iteration {t} loss: {loss.item()}")

    return model


def do_model(nnmodel, seed):
    np.random.seed = seed
    p = np.random.randint(30, size=24)
    m = gp.Model()
    x = m.addMVar((1, 48), vtype=GRB.CONTINUOUS, name="x", lb=0, ub=410)
    y = m.addMVar((1, 24), lb=20, ub=30, vtype=GRB.CONTINUOUS, name="y")
    m.setObjective(p @ x[0, 24:], GRB.MINIMIZE)

    nn2gurobi = Sequential(m, nnmodel, x, y)

    return (m, nn2gurobi)


def gen_nn(seed):
    filename = "{}-seed{}.pkl".format("Kadir_torch", seed)
    try:
        torch.load(filename)
        return
    except Exception:
        pass
    model = do_regression(seed)
    torch.save(model, filename)


def heuristic(nn, nn2gurobi):
    X = torch.from_numpy(np.genfromtxt("X.csv")).float()

    prediction = nn.forward(X)
    feasibles = X[((prediction >= 20) & (prediction <= 30)).all(axis=1), :]
    sortedinputs = np.argsort(nn2gurobi._layers[0].invar.Obj @ feasibles.numpy().T)

    prop(nn2gurobi, feasibles[sortedinputs[0, 0]].numpy().reshape(1, -1), reset=True)


def doone(filename, doobbt=None, doheuristic=None, seed=None):
    outputfile = filename.strip(".joblib") + f"-objseed{seed}.lp.bz2"
    try:
        with gp.read(outputfile) as m:
            return
    except Exception:
        pass

    model = torch.load(f"Networks_pytorch/{filename}")

    if filename.startswith("Kadir"):
        m, nn2gurobi = do_model(model, seed)
    else:
        return
    if doobbt:
        nn2gurobi.obbt(1)
    if doheuristic:
        heuristic(model, nn2gurobi)
        m.write(outputfile[: -len("lp.bz2")] + "attr")
    m.write(outputfile)


def gen_all_nn():
    with parallel_backend("multiprocessing"):
        do = gen_nn
        Parallel(n_jobs=8, verbose=10)(delayed(do)(seed) for seed in range(10))


if __name__ == "__main__":
    files = [f for f in os.listdir("Networks_pytorch") if f.startswith("Kadir") and f.endswith(".pkl")]
    doobbt = False
    doheuristic = False

    Parallel(n_jobs=1, verbose=10)(delayed(doone)(f, doobbt, doheuristic, seed) for f in files for seed in range(1, 11))
