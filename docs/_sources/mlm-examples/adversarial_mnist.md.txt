---
jupytext:
  formats: ipynb///ipynb,myst///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Adversarial Machine Learning

In this example, we show how to use Gurobi Machine Learning to construct an adversarial example for a trained neural network.

We use the MNIST handwritten digit database (http://yann.lecun.com/exdb/mnist/) for this example.


## Introduction

For this problem, we are given a trained neural network and one well classified example $\bar x$.
Our goal is to construct another example $x$ _close to_ $\bar x$ that is classified with another label.

For the hand digit recognition problem, the input is a grayscale image of $28 \times 28$ ($=784$) pixels and the output is a vector of length 10 (each entry corresponding to a digit). We denote the output vector by $y$. The image is classified according to the largest entry of $y$.

For the training example, assume that coordinate $l$ of the output vector was the one with the largest value giving the correct label. We pick a coordinate corresponding to another label, denoted $w$, and we want the difference between $y_w - y_l$ to be as large as possible.

If we can find a solution where this difference is positive, then $x$ is a counter-example that will receive a different label. If instead we can show that the difference is non-positive, no such counter-example exists.

In this example, we define the neighborhood using the $l1-$norm $|| x - \bar x ||_1$. The size of the neighborhood is defined by a fixed parameter $\delta$. We want

$$
|| x - \bar x ||_1 \le \delta.
$$

If we denote by $g$ the prediction function of the neural network. Our full optimization model reads:

$$
\begin{aligned}
&\max y_w - y_l \\
&\text{subject to:}\\
&|| x - \bar x ||_1 \le \delta,\\
& y = g(x).
\end{aligned}
$$


Our model is inspired by <cite data-cite="fischetti_jo_2018">Fischet al. (2018)</cite>.

## Imports and loading data

First, we import the required packages for this example.

In addition to the usual packages, we will need `matplotlib` to plot the digits, `joblib` to load a pre-trained network and `pandas` that is the format in which training examples are stored.

Note that from the `gurobi_ml` package we need to use directly the `add_mlp_regressor_constr` function for reasons that will be clarified later.

```{code-cell}
import gurobipy as gp
import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt

from gurobi_ml.sklearn import add_mlp_regressor_constr
```

Now we can load the data.

We load a neural network that was pre-trained with Scikit-learn MLPRegressor. The network is quite small (2 hidden layers of 50 neurons), finding counter example shouldn't be too difficult.

We also load the first 100 training examples of the MNIST dataset that we saved to avoid having to reload the full data set.

```{code-cell}
# Load the trained network and the examples
nn = load("../../../tests/predictors/MNIST_50_50.joblib")
X = load("../../../tests/predictors/MNIST_first100.joblib")
```

## Choose an example and set labels

Now we choose an example. Here we chose arbitrarily example 26.
We plot, the example and verify it is well predicted by calling the `predict` function.

```{code-cell}
# Choose an example
exampleno = 26
example = X.iloc[exampleno : exampleno + 1, :]

pixels = example.to_numpy().reshape((28, 28))
plt.imshow(pixels, cmap="gray")
plt.show()

print(f"Predicted label {nn.predict(example)}")
```

To set up the objective function of the optimization model, we need to find a wrong label.

We use `predict_proba` to get the weight for each label given by the neural network.
We then use `numpy`'s `argsort` function to get the labels sorted by their weight.
The right label is then the last element in the list, and we pick the next to last element as the wrong label.

```{code-cell}
ex_prob = nn.predict_proba(example)
sorted_labels = np.argsort(ex_prob)[0]
right_label = sorted_labels[-1]
wrong_label = sorted_labels[-2]
```

## Building the optimization model

Now that all the data is gathered we can proceed to building the optimization model.

We need to create a matrix variable `x` corresponding to the new input of the neural network we want to compute and a `y` variables for the output of the neural network. Those variables should have respectively the shape of the example we picked and the shape of the return value of `predict_proba`.

We also need additional variables to model the $l1$-norm constraint. Namely, for each pixel in the image, we need to model the absolute difference between $x$ and $\bar x$. We will detail the model below. For now, we create the necessary variables that are another matrix of variable of same shape as `x`.

We also set the objective which is to maximize the difference between the _wrong_ label and the _right_ label.

```{code-cell}
m = gp.Model()
delta = 5

x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")
y = m.addMVar(ex_prob.shape, lb=-gp.GRB.INFINITY, name="y")

abs_diff = m.addMVar(example.shape, lb=0, ub=1, name="abs_diff")

m.setObjective(y[0, wrong_label] - y[0, right_label], gp.GRB.MAXIMIZE)
```

The $l1-$norm constraint is formulated with:

$$
\eta \ge x - \bar x \\
\eta \ge \bar x - x \\
\sum \eta \le \delta
$$

With $\eta$ denoting the `absdiff` variables.

Those constraints are naturally expressed with Gurobi Matrix API. Note that we need to convert the example which is a `pandas.DataFrame` to `numpy` for the matrix API.

```{code-cell}
# Bound on the distance to example in norm-1
m.addConstr(abs_diff >= x - example.to_numpy())
m.addConstr(abs_diff >= -x + example.to_numpy())
m.addConstr(abs_diff.sum() <= delta)

# Update the model
m.update()
```

Finally, we insert the neural network in the `gurobipy` model to link `x` and `y`.

Note that this case is not as straightforward as others. The reason is that the neural network is trained for classification with a `"softmax"` activation in the last layer. But in this model we are using the network without activation in the last layer.

For this reason, we change manually the last layer activation before adding the network to the Gurobi model.

Also, we use the function `add_mlp_regressor_constr` directly. The network being actually for classification (i.e. of type `MLPClassifier`, the `add_predictor_constr` function would not handle it automatically.

Note that in the output, you should see a warning about adding constraints with very small coefficients that are ignored. Neural-networks often contains very small coefficients in their expression. Any coefficient with an absolute value smaller than $10^-{13}$ is ignored by Gurobi. This may result in slightly different predicted values but should be negligible.

```{code-cell}
# Change last layer activation to identity
nn.out_activation_ = "identity"
# Code to add the neural network to the constraints
pred_constr = add_mlp_regressor_constr(m, nn, x, y)

# Restore activation
nn.out_activation_ = "softmax"
```

The model should be complete. We print the statistics of what was added to insert the neural network into the optimization model.

```{code-cell}
pred_constr.print_stats()
```

## Solving the model

We now turn to solving the optimization model.
Solving the adversarial problem, as we formulated it above, doesn't actually require computing a provably optimal solution. Instead, we need to either:

   - find a feasible solution with a positive objective cost (i.e. a counter-example), or
   - prove that there is no solution of positive cost (i.e. no counter-example in the neighborhood).

 We can use Gurobi parameters to limit the optimization to answer those questions: setting BestObjStop (https://www.gurobi.com/documentation/current/refman/bestobjstop.html#parameter:BestObjStop) to 0.0 will stop the optimizer if a counter-example is found, setting BestBdStop (https://www.gurobi.com/documentation/current/refman/bestobjstop.html#parameter:BestObjStop) to 0.0 will stop the optimization if the optimizer has shown there is no counter-example.

We set the two parameters and optimize.

```{code-cell}
m.Params.BestBdStop = 0.0
m.Params.BestObjStop = 0.0
m.optimize()
```

## Results

Normally for the example and $\delta$ we chose a counter example is found. We finish this notebook by plotting the example and printing how it is classified by the neural network.

```{code-cell}
pixels = x.X.reshape((28, 28))
plt.imshow(pixels, cmap="gray")
plt.show()

example_mod = pd.DataFrame(data=x.X, columns=example.columns, index=example.index)
print(f"Solution is classified as {nn.predict(example_mod)}")
```

+++ {"tags": ["hide-input"]}

Copyright © 2022 Gurobi Optimization, LLC
