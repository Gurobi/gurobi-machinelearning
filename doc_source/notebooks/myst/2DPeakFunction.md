---
jupytext:
  formats: ipynb///ipynb,myst///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: gurobi-ml
  language: python
  name: Python3
---

# Surrogate functions

Extra required packages:
- matplotlib

<cite data-cite="Henao_Maravelias_2011"></cite>
<cite data-cite="Lu_Pu_2017"></cite>
<cite data-cite="Henao_Maravelias_2011"></cite>
Optimization solvers often struggle to prove a global optimum of a model when
it holds highly nonlinear functions. The reason is often that relaxations
of the nonlinear functions are not good enough to make the solver prove an
acceptable bound in a reasonable amount of time. Another issue might be that a
given solver does not support nonlinear functions, but only accepts linear ones.

One possible solution for this problem is to approximate the problematic nonlinear
functions via neural networks and use MIP technology to solve the constructed
approximation efficiently. A piecewise-linear approximation of the nonlinear
function of interest might be considered but usually gets more expensive and hard to
model with increasing function dimensions. In the following example, we show how to
approximate a nonlinear function via sklearn's MLPRegressor and accordingly solve the
neural network approximation of the nonlinear function with Gurobi.

The purpose of this example is solely to present the idea of approximating
any given function via neural networks and solving the approximation through MIP
technology.

The function we will approximate is the 2D peak function which can be found on
many mathematical company logos and book covers. The function is given as

$$
f(x,y) = 3 \cdot (1-x)^2 \cdot \exp(-x^2 - (y+1)^2) - 10
         \cdot (\frac{x}{5} - x^3 - y^5) \cdot \exp(-x^2 - y^2) -
         \frac{1}{3} \cdot \exp(-(x+1)^2 - y^2).
$$

In this example, we will search for the global minimum of %f% over a small interval.

$$
\begin{aligned}
&\min_{x,y} f(x,y)\\
&\text{s.t.}\\
&x,y \in [-1,1]
\end{aligned}
$$

To find the global minimum of $f$, we will approximate $f(x,y)$ through a neural
network function $g(x,y)$ to obtain a MIP and solve

$$
\begin{aligned}
&\min_{x,y} g(x,y) \approx f(x,y)\\
&\text{s.t.}\\
&x,y \in [-1,1]
\end{aligned}
$$

with Gurobi.

```{code-cell}
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from gurobi_ml.sklearn import PipelineConstr
```

##### Define the nonlinear function of interest

```{code-cell}
def peak2d(xx, yy):
    return (
        3 * (1 - xx) ** 2.0 * np.exp(-(xx**2) - (yy + 1) ** 2)
        - 10 * (xx / 5 - xx**3 - yy**5) * np.exp(-(xx**2) - yy**2)
        - 1 / 3 * np.exp(-((xx + 1) ** 2) - yy**2)
    )
```

```{code-cell}
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
xx, yy = np.meshgrid(x, y)
z = peak2d(xx, yy)
```

```{code-cell}
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm,
                       linewidth=0.01, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```

```{code-cell}
X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
y = z.ravel()
```

##### Approximate the function and define a test set

```{code-cell}
# Run our regression
layers = [30, 30, 30]
regression = MLPRegressor(hidden_layer_sizes=layers, activation="relu")
pipe = make_pipeline(PolynomialFeatures(), regression)
pipe.fit(X=X, y=y)
```

```{code-cell}
X_test = np.random.random((100, 2)) * 2 - 1
```

```{code-cell}
metrics.r2_score(peak2d(X_test[:, 0], X_test[:, 1]), pipe.predict(X_test))
```

```{code-cell}
metrics.max_error(peak2d(X_test[:, 0], X_test[:, 1]), pipe.predict(X_test))
```

The maximum error is quite high but still acceptable for the purpose of this example.

```{code-cell}
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(
    xx,
    yy,
    pipe.predict(X).reshape(xx.shape),
    cmap=cm.coolwarm,
    linewidth=0.01,
    antialiased=False,
)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```

Optically, the approximation looks very close to the original function.

+++

##### Construct the optimization model

```{code-cell}
optfeat = [0, 1]
```

```{code-cell}
# Start with classical part of the model
m = gp.Model()

x = m.addMVar((len(optfeat)), lb=-1, ub=1, name="x")
y = m.addVar(lb=-GRB.INFINITY, name="y")

m.setObjective(y, gp.GRB.MINIMIZE)

# create transforms to turn scikit-learn pipeline into Gurobi constraints
PipelineConstr(m, pipe, x, y)
```

##### Finally optimize the model

```{code-cell}
m.Params.TimeLimit = 20
m.Params.MIPGap = 0.1
m.Params.NonConvex = 2
```

```{code-cell}
m.optimize()
```

```{code-cell}
m.NumVars
```

##### Look at the solution and objective value

```{code-cell}
print(x.X) # solution point of the approximated problem
print(m.ObjVal) # objective value of the approximated problem
print(peak2d(x[0].X, x[1].X)) # function value at the solution point of the approximated problem
print(abs(peak2d(x[0].X, x[1].X) - m.ObjVal)) # error of the approximation
```

The difference between the function and the approximation at the computed solution point is noticable but acceptable given the big maximum error of the neural network approximation.

+++

Copyright © 2020 Gurobi Optimization, LLC
