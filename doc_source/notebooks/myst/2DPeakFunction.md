---
jupytext:
  formats: ipynb///ipynb,myst///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Surrogate Models

Some industrial applications require modeling complex processes
that can result either in highly non-linear functions or functions
defined by a simulation process.
In those contexts, optimization solvers often struggle.
The reason may be that relaxations
of the nonlinear functions are not good enough to make the solver prove an
acceptable bound in a reasonable amount of time. Another issue may be that
the solver is not able to represent the functions.

An approach that has been proposed in the literature is
to approximate the problematic nonlinear
functions via neural networks with ReLU activation and use MIP technology
to solve the constructed approximation
(see for e.g. <cite data-cite="Henao_Maravelias_2011"></cite>, <cite data-cite="Schweidtmann_2022"></cite>).
This use of neural network can be motivated by their ability to provide a
universal approximation (see for e.g. {<cite data-cite="Lu_Pu_2017"></cite>).
This use of ML models to replace complex processes is often referred to as *surrogate models*.

In the following example, we
approximate a nonlinear function via `Scikit-learn` `MLPRegressor` and then to solve an
optimization problem that uses the approximation of the nonlinear function with Gurobi.

The purpose of this example is solely illustrative and doesn't
relate to any particular application.

The function we approximate is the [2D peaks function](https://www.mathworks.com/help/matlab/ref/peaks.html#mw_46aeee28-390e-4373-aa47-e4a52447fc85).

The function is given as

$$
\begin{aligned}
f(x,y) = & 3 \cdot (1-x)^2 \cdot \exp(-x^2 - (y+1)^2) - \\
         & 10 \cdot (\frac{x}{5} - x^3 - y^5) \cdot \exp(-x^2 - y^2) - \\
         & \frac{1}{3} \cdot \exp(-(x+1)^2 - y^2).
\end{aligned}
$$

In this example, we want to find the minimum of $f$ over the interval $[-2, 2]$.

$$
\begin{aligned}
&\min_{x,y} f(x,y)\\
&\text{s.t.}\\
&x,y \in [-2,2].
\end{aligned}
$$

The global minimum of this problem can be found numerically
to have value $-6.55113$ at the point $(0.2283, -1.6256)$
(see for example [here](https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/talks.d/t09talks.d/09waterloomatlab.d/optimTipsWebinar/html/optimTipsTricksWalkthrough.html#18)).

Here to find this minimum of $f$, we approximate $f(x,y)$ through a neural
network function $g(x,y)$ to obtain a MIP and solve

$$
\begin{aligned}
&\min_{x,y} g(x,y) \approx f(x,y)\\
&\text{s.t.}\\
&x,y \in [-2,2].
\end{aligned}
$$

First import the necessary packages. Before applying the neural network,
we do a preprocessing to extract polynomial features of degree 2. Hopefully this
will help us to approximate the smooth function.
Besides, `gurobipy`, `numpy` and the appropriate
`sklearn` objects, we also use `matplotlib` to plot the function, and it's approximation.

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
from gurobi_ml import add_predictor_constr
```

## Define the nonlinear function of interest

We define the 2D peak function as a python function.

```{code-cell}
def peak2d(xx, yy):
    return (
        3 * (1 - xx) ** 2.0 * np.exp(-(xx**2) - (yy + 1) ** 2)
        - 10 * (xx / 5 - xx**3 - yy**5) * np.exp(-(xx**2) - yy**2)
        - 1 / 3 * np.exp(-((xx + 1) ** 2) - yy**2)
    )
```

To train the neural network, we make a uniform sample of the domain of the
function in the region of interest using `numpy`'s `meshgrid` function.

We then plot the function with `matplotlib`

```{code-cell}
x = np.arange(-2, 2, 0.01)
y = np.arange(-2, 2, 0.01)
xx, yy = np.meshgrid(x, y)
z = peak2d(xx, yy)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm,
                       linewidth=0.01, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```

## Approximate the function

To fit a model, we need to reshape our data. We concatenate the values of `x` and `y` in
an array `X` and make `z` one dimensional.

```{code-cell}
X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
z = z.ravel()
```

To approximate the function, we use a `Pipeline` with polynomial features and
a neural-network regressor. We do a relatively small neural-network.

```{code-cell}
# Run our regression
layers = [30]*2
regression = MLPRegressor(hidden_layer_sizes=layers, activation="relu")
pipe = make_pipeline(PolynomialFeatures(), regression)
pipe.fit(X=X, y=z)
```

To test the accuracy of the approximation, we take a random sample of points, and
we print the $R^2$ value and the maximal error.

```{code-cell}
X_test = np.random.random((100, 2)) * 4 - 2

r2_score = metrics.r2_score(peak2d(X_test[:, 0], X_test[:, 1]), pipe.predict(X_test))
max_error = metrics.max_error(peak2d(X_test[:, 0], X_test[:, 1]), pipe.predict(X_test))
print("R2 error {}, maximal error {}".format(r2_score, max_error))
```

While the $R^2$ value is good, the maximal error is quite high. For the purpose of this
example we still deem it acceptable. We plot the function.

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

Visually, the approximation looks close enough to the original function.

## Build and Solve Optimization Model

We now turn to the optimization model. For this model we want to find the minimal value
of $y$.

Note that in this simple example, we don't use matrix variables but regular Gurobi variables
instead.

```{code-cell}
m = gp.Model()

x = m.addVars(2, lb=-2, ub=2, name="x")
y = m.addVar(lb=-GRB.INFINITY, name="y")

m.setObjective(y, gp.GRB.MINIMIZE)

# add "surrogate constraint"
pred_constr = add_predictor_constr(m, pipe, x, y)

pred_constr.print_stats()
```

Now call `optimize`. Since we use polynomial features the resulting model is a
non-convex quadratic problem. In Gurobi, we need to set the parameter `NonConex`
to 2 to be able to solve it.

```{code-cell}
m.Params.TimeLimit = 20
m.Params.MIPGap = 0.1
m.Params.NonConvex = 2

m.optimize()
```

After solving the model, we check the error in the estimate of the Gurobi solution.

```{code-cell}
print("Error in approximating the regression {:.6}".format(np.max(np.abs(pred_constr.get_error()))))
```

Finally, we look at the solution and the objective value
found.

```{code-cell}
print(f"solution point of the approximated problem ({x[0].X:.4}, {x[1].X:.4})" +
        f"Objective value {m.ObjVal}.")
print(f"Function value at the solution point {peak2d(x[0].X, x[1].X)} error {abs(peak2d(x[0].X, x[1].X) - m.ObjVal)}.")
```

The difference between the function and the approximation at the computed solution point is noticeable, but the point we found is reasonably close to the actual global minima.
Depending on the use case this might be deemed acceptable. Of course, training a larger network should result in a better approximation.

Copyright © 2020 Gurobi Optimization, LLC
