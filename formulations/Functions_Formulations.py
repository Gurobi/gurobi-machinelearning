import gurobipy as gp
import numpy as np
from gurobipy import GRB
from joblib import Parallel, delayed, dump, load, parallel_backend
from ml2gurobi.sklearn import Pipe2Gurobi
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline


def GoldsteinPrice(x1, x2):
    fact1a = (x1 + x2 + 1) ** 2
    fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    fact2 = 30 + fact2a * fact2b
    return fact1 * fact2


def peak2d(xx, yy):
    return (
        3 * (1 - xx) ** 2.0 * np.exp(-(xx**2) - (yy + 1) ** 2)
        - 10 * (xx / 5 - xx**4 - yy**5) * np.exp(-(xx**2) - yy**2)
        - 1 / 3 * np.exp(-((xx + 1) ** 2) - yy**2)
    )


def nnapprox2dfunc(function, layers, random_state):
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    xx, yy = np.meshgrid(x, y)
    z = function(xx, yy)

    X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
    y = z.ravel()

    regression = MLPRegressor(hidden_layer_sizes=layers, random_state=random_state, activation="relu")
    pipe = make_pipeline(regression)
    pipe.fit(X=X, y=y)
    return pipe


def do_model(pipe, reluformulation=None):
    optfeat = [0, 1]

    m = gp.Model()

    x = m.addMVar((1, len(optfeat)), lb=-1, ub=1, name="x")
    y = m.addMVar((1, 1), lb=-GRB.INFINITY, name="y")

    m.setObjective(y.sum(), gp.GRB.MINIMIZE)

    pipe2gurobi = Pipe2Gurobi(pipe, m)
    if reluformulation is not None:
        pipe2gurobi.steps[-1].actdict["relu"] = reluformulation
    pipe2gurobi.predict(X=x, y=y)

    m._pipe2gurobi = pipe2gurobi
    return m


def gen_nn(function, layers, seed):
    filename = "{}_nn-{}_seed{}.joblib".format(function.__name__, "-".join([f"{n}" for n in layers]), seed)
    try:
        with load(filename):
            return
    except Exception:
        pass
    pipe = nnapprox2dfunc(function, layers, seed)
    nn = pipe.steps[-1][1]
    for layer in nn.coefs_:
        layer[np.abs(layer) < 1e-8] = 0.0
    dump(pipe, filename)


def gen_all_nn():
    do = gen_nn
    with parallel_backend("multiprocessing"):
        Parallel(n_jobs=4, verbose=10)(
            delayed(do)(peak2d, hidden_layers, seed)
            for hidden_layers in (
                [56] * 2,
                [56] * 3,
                [128] * 2,
                [128] * 3,
                [256] * 2,
                [256] * 3,
            )
            for seed in range(10)
        )

        Parallel(n_jobs=4, verbose=10)(
            delayed(do)(GoldsteinPrice, hidden_layers, seed)
            for hidden_layers in (
                [56] * 2,
                [56] * 3,
                [128] * 2,
                [128] * 3,
                [256] * 2,
                [256] * 3,
            )
            for seed in range(10)
        )


if __name__ == "__main__":
    gen_all_nn()
