from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import load
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import gurobipy as gp
from gurobipy import GRB

import sys
sys.path.append('..')
from ml2grb.sklearn2grb import Pipe2Gurobi
from ml2grb.extra.morerelu import ReLUmin
from ml2grb.activations2grb import Identity


def do_formulation(pipe, X, exampleno, filename, doobbt=0, docuts=False):
    example = X[exampleno:exampleno+1, :]
    ex_prob = pipe.predict_proba(example)
    output_shape = ex_prob.shape

    sortedidx = np.argsort(ex_prob)[0]

    m = gp.Model()
    epsilon = 5
    lb = np.maximum(example - epsilon, 0)
    ub = np.minimum(example + epsilon, 1)

    x = m.addMVar(example.shape, lb=lb, ub=ub, name='X')
    absdiff = m.addMVar(example.shape, lb=0, ub=1, name='dplus')
    output = m.addMVar(output_shape, lb=-gp.GRB.INFINITY, name='y')

    m.addConstr(absdiff[0, :] >= x[0, :] - example[0, :])
    m.addConstr(absdiff[0, :] >= - x[0, :] + example[0, :])
    m.addConstr(absdiff[0, :].sum() <= epsilon)
    pipe2grb = Pipe2Gurobi(pipe, m)
#    pipe2grb.steps[-1].actdict['relu'] = ReLUmin()
    pipe2grb.steps[-1].actdict['softmax'] = Identity()
    pipe2grb.predict(x, output)
    m.setObjective(output[0, sortedidx[0]] -
                   output[0, sortedidx[-1]], gp.GRB.MAXIMIZE)
    m.update()

    m.Params.OutputFlag = 1
    try:
        if doobbt:
            pipe2grb.steps[0].obbt(doobbt)
    except:
        return
    m.update()
    m.write(filename)

def doone(filename, exampleno, docuts, doobbt):
    outputfile = filename.strip('.joblib') + f'-{exampleno}.lp.bz2'
    try:
        with gp.read(outputfile) as m:
            return
    except:
        pass
    pipe = load(filename)
    X = load('MNIST_first100.joblib')
    do_formulation(pipe, X, exampleno, outputfile, doobbt, docuts)

if __name__ == "__main__":
    files = ('MNIST_50_50.joblib', 'MNIST_100_100.joblib')
    docuts = False
    doobbt = False

    r = Parallel(n_jobs=4, verbose=10)(
        delayed(doone)(f, n, docuts, doobbt)
        for f in files for n in range(100))
