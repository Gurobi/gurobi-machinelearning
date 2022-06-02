import os

from joblib import dump, load
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml2gurobi.sklearn import (
    decisiontreeregressor2gurobi,
    gradientboostingregressor2gurobi,
    linearregression2gurobi,
    mlpregressor2gurobi,
)


class DiabetesCases:
    to_test = [(LinearRegression(), linearregression2gurobi),
               (DecisionTreeRegressor(max_leaf_nodes=50), decisiontreeregressor2gurobi),
               (GradientBoostingRegressor(n_estimators=20), gradientboostingregressor2gurobi),
               (MLPRegressor([20, 20]), mlpregressor2gurobi)]

    def __init__(self):
        self.basedir = os.path.join(os.path.dirname(__file__), 'predictors')
        version = None
        with open(os.path.join(self.basedir, 'sklearn_version')) as filein:
            version = filein.read()
        if version != sklearn_version:
            print("Scikit learn version changed. Regenerate predictors")
            self.build_predictors()
            with open(os.path.join(self.basedir, 'sklearn_version'), 'w') as fileout:
                fileout.write(sklearn_version)

    def build_predictors(self):
        data = datasets.load_diabetes()

        X = data['data']
        y = data['target']

        for predictor, _ in self.to_test:
            predictor.fit(X, y)
            filename = f'diabetes_none_{type(predictor).__name__}.joblib'
            rval = {'predictor': predictor,
                    'input_shape': X.shape,
                    'output_shape': y.shape}

            dump(rval, os.path.join(self.basedir,filename))

        for predictor, _ in self.to_test:
            pipeline = make_pipeline(StandardScaler(), predictor)
            pipeline.fit(X, y)
            filename = f'diabetes_pipe_{type(predictor).__name__}.joblib'
            rval = {'predictor': pipeline,
                    'input_shape': X.shape,
                    'output_shape': y.shape}

            dump(rval, os.path.join(self.basedir,filename))

    def get_case(self, predictor, withpipe):
        if withpipe:
            withpipe = 'pipe'
        else:
            withpipe = 'none'
        filename = f'diabetes_{withpipe}_{type(predictor).__name__}.joblib'
        predictor = load(os.path.join(self.basedir, filename))
        return predictor
