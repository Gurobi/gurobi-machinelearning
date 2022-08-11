"""
Created on Mon Jul 19 17:26:18 2021

@author: 4ka
"""

import os
import pprint
import time

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB

from gurobi.machinelearning.nnalgs import prop
from gurobi.machinelearning.pytorch import Sequential2Gurobi

# Load data
X = torch.from_numpy(np.genfromtxt("X.csv")).float()
Y = torch.from_numpy(np.genfromtxt("Y.csv")).float()


def heuristic(modelname, seed):
    model = torch.load(modelname)

    np.random.seed = seed
    p = np.random.randint(30, size=24)

    m = gp.Model()
    x = m.addMVar((1, 48), vtype=GRB.CONTINUOUS, name="x", lb=0, ub=410)
    y = m.addMVar((1, 24), lb=20, ub=30, vtype=GRB.CONTINUOUS, name="y")
    m.setObjective(p @ x[0, 24:], GRB.MINIMIZE)
    m.update()

    # Add constraint to predict value of y using x
    nn2gurobi = Sequential2Gurobi(model, m, clean_regressor=True)
    nn2gurobi.predict(x, y)
    m.Params.OutputFlag = 0
    m.Params.Threads = 4

    print(f"Starting heuristic on {modelname} obj seed {seed}")

    tottime = -time.monotonic()
    prediction = model.forward(X)
    feasibles = X[((prediction >= 20) & (prediction <= 30)).all(axis=1), :]
    sortedinputs = np.argsort(nn2gurobi._layers[0].invar.Obj @ feasibles.numpy().T)

    xin = feasibles[sortedinputs[0, 0]].numpy().reshape(1, -1)

    for it in range(100):
        prop(nn2gurobi, xin, reset=False)

        for v in nn2gurobi.canrelax:
            v.LB = 0.0
            v.UB = 1.0
        m.update()
        nn2gurobi.obbt(1)

        m.Params.OutputFlag = 1
        m.optimize()

        xin = x.X
        obj = m.ObjVal
        print(f"Iteration {it} obj: {obj:.2f}")
        if m.ObjVal < 1e-2:
            break
        for layer in nn2gurobi._layers:
            layer.zvar.UB = 1.0
            layer.zvar.LB = 0.0
    tottime += time.monotonic()
    print()
    print(f"Heuristic finished value {obj} time {tottime:.2f}")
    print()
    m.dispose()
    return {"status": "Worked", "time": tottime, "obj": obj}


if __name__ == "__main__":
    results = dict()
    files = ["Networks_pytorch/" + f for f in os.listdir("Networks_pytorch") if f.startswith("Kadir") and f.endswith(".pkl")]
    for f in files:
        for i in range(1, 11):
            try:
                results[(f, i)] = heuristic(f, i)
            except Exception:
                results[(f, i)] = {"Status": "Failed"}

    pprint.pprint(results)
