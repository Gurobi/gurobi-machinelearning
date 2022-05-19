# read version from installed package
__version__ = "0.1.0"

# Populate package namespace
from .sklearn import (
    DecisionTree2Gurobi,
    GradientBoostingRegressor2Gurobi,
    LinearRegression2Gurobi,
    MLPRegressor2Gurobi,
    Pipe2Gurobi,
    StandardScaler2Gurobi,
)

try:
    import torch

    from .pytorch import Sequential2Gurobi
except ImportError as e:
    pass
