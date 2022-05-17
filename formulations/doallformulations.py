import os

import gurobipy as gp
from Functions_Formulations import do_model as func_model
from Janos_Formulations import dojanosformulation as janos_model
from joblib import Parallel, delayed, load
from Kadir_formulation import do_model as kadir_model


def doone(filename, docuts=None, doobbt=None, reluactivation=None):
    outputfile = filename.strip('.joblib') + '.lp.bz2'
    try:
        with gp.read(outputfile) as m:
            return
    except Exception:
        pass

    pipe = load(f'../Networks/{filename}')

    try:
        if filename.startswith('Kadir'):
            m = kadir_model(pipe)
        elif filename.startswith('Janos'):
            m = janos_model(pipe)
        else:
            m = func_model(pipe)
        if doobbt:
            m._pipe2gurobi.steps[-1].obbt()
    except Exception:
        print(f'Failed on {filename}')
        raise

    m.write(outputfile)


if __name__ == "__main__":
    files = os.listdir('../Networks')
    docuts = False
    doobbt = False

    r = Parallel(n_jobs=4, verbose=10)(delayed(doone)(f, docuts, doobbt) for f in files)
