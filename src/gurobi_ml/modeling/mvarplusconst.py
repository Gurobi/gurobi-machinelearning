# Copyright © 2022 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gurobipy as gp
import numpy as np


class MVarPlusConst:
    __array_priority__ = 100

    def __init__(self, constant, var_index, mvar):
        self.constant = constant
        self.var_index = var_index
        self.mvar = mvar

    @staticmethod
    def fromdf(df):
        array = df.to_numpy()
        const = np.zeros(df.shape)
        mvar_idx = []
        for i, a in enumerate(array.T):
            if all(map(lambda i: isinstance(i, gp.Var), a)):
                mvar_idx.append(i)
            else:
                try:
                    const[:, i] = a.astype(np.float64)
                except TypeError:
                    raise TypeError(
                        "Dataframe can't be converted because of mixed types"
                    )
        if len(mvar_idx) > 0:
            vars = gp.MVar.fromlist(array[:, mvar_idx].tolist())
        else:
            vars = 0
        return MVarPlusConst(const, mvar_idx, vars)

    def tomlinexpr(self):
        r_val = gp.MLinExpr.zeros(self.constant.shape)
        r_val += self.constant
        r_val[:, self.var_index] += self.mvar
        return r_val

    def __repr__(self):
        return self.tomlinexpr().__repr__()

    def __add__(self, B):
        if isinstance(B, (gp.MLinExpr, gp.MVar)):
            r_val = gp.MLinExpr.zeros(self.constant.shape)
            r_val += self.constant + B
            r_val[:, self.var_index] += self.mvar
        else:
            B = B.astype(float)
            constant = self.constant + B
            r_val = MVarPlusConst(constant, self.var_index, self.mvar)

        return r_val

    def __radd__(self, B):
        if isinstance(B, (gp.MLinExpr, gp.MVar)):
            r_val = gp.MLinExpr.zeros(self.constant.shape)
            r_val += self.constant + B
            r_val[:, self.var_index] += self.mvar
        else:
            B = B.astype(float)
            constant = self.constant + B
            r_val = MVarPlusConst(constant, self.var_index, self.mvar)

        return r_val

    def __mul__(self, B):
        r_val = gp.MLinExpr.zeros(self.constant.shape)
        r_val += self.constant * B
        if len(B.shape) == 2 and B.shape[1] == self.constant.shape[1]:
            r_val[:, self.var_index] += self.mvar * B[:, self.var_index]
        else:
            r_val[:, self.var_index] += self.mvar * B
        return r_val

    def __rmul__(self, B):
        r_val = gp.MLinExpr.zeros(self.constant.shape)
        r_val += self.constant * B
        if len(B.shape) == 2 and B.shape[1] == self.constant.shape[1]:
            r_val[:, self.var_index] += self.mvar * B[:, self.var_index]
        else:
            r_val[:, self.var_index] += self.mvar * B
        return r_val

    def __matmul__(self, B):
        if len(B.shape) == 2:
            r_val = gp.MLinExpr.zeros((self.constant.shape[0], B.shape[1]))
            dim = 2
        elif len(B.shape) == 1:
            r_val = gp.MLinExpr.zeros((self.constant.shape[0]))
            dim = 1
        else:
            raise ValueError("Right operand has too many dimensions")
        r_val += self.constant @ B
        if dim == 2:
            r_val += self.mvar @ B[self.var_index, :]
        else:
            r_val += self.mvar @ B[self.var_index]
        return r_val

    def __getitem__(self, obj):
        if not isinstance(obj, tuple):
            raise ValueError("Only deal with tuples")
        if len(obj) != 2:
            raise ValueError("Only deal with tuples of size 2")
        if isinstance(obj[1], slice):
            raise Exception("Dealing with slices not implemented")
        if not isinstance(obj[1], (int, np.longlong)):
            print(obj[1])
            print(type(obj[1]))
            raise Exception("Dealing with anything else than int is not implemented")
        if obj[1] in self.var_index:
            return self.mvar[obj[0], self.var_index.index(obj[1])]
        # Oh that's a dirty trick
        # For the decision tree formulation, we need a constant right hand side of an
        # indicator... seems to work with a constant LinExpr
        return self.constant[obj]

    @property
    def shape(self):
        return self.constant.shape

    def get_value(self):
        r_val = self.constant.copy()
        r_val[:, self.var_index] += self.mvar.X
        return r_val

    def __rmatmul__(self, B):
        if len(B.shape) == 2:
            r_val = gp.MLinExpr.zeros((B.shape[0], self.constant.shape[1]))
            dim = 2
        elif len(B.shape) == 1:
            r_val = gp.MLinExpr.zeros((B.shape[0]))
            dim = 1
        else:
            raise ValueError("Right operand has too many dimensions")
        r_val += B @ self.constant
        if dim == 2:
            r_val[:, self.var_index] += B @ self.mvar
        else:
            r_val[:, self.var_index] += B @ self.mvar
        return r_val
