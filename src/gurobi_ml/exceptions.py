# Copyright © 2022 Gurobi Optimization, LLC


class NotRegistered(Exception):
    def __init__(self, predictor):
        super().__init__("Object of type {} is not registered/supported with gurobi_ml".format(type(predictor).__name__))


class NoModel(Exception):
    def __init__(self, predictor, reason):
        super().__init__("Can't do model for {}: {}".format(type(predictor).__name__, reason))


class NoSolution(Exception):
    def __init__(self):
        super().__init__("No solution available")


class ModelingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class InternalError(Exception):
    def __init__(self, message):
        super().__init__(message)
