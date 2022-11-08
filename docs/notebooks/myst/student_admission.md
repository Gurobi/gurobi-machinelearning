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

# Student Enrollment

In this example, we show how to reproduce the model of student enrollment from
<cite data-cite="JANOS">Bergman et.al. (2020)</cite> with Gurobi Machine
Learning.

This model was developed in the context of the development of
[Janos](https://github.com/INFORMSJoC/2020.1023), a toolkit similar to Gurobi
Machine Learning to integrate ML models and Mathematical Optimization.

This example illustrates in particular how to use the logistic regression and
tune the piecewise-linear approximation of the logistic function.

We also show how to deal with fixed features in the optimization model using
pandas data frames.

In this model, data of students admissions in a college is used to predict the
probability that a student enrolls to the college.

The data has 3 features: the SAT and GPA scores of each student, and the
scholarship (or merit) that was offered to each student. Finally, it is known if
each student decided to join the college or not.

Based on this data a logistic regression is trained to predict the probability
that a student joins the college.

Using this regression model, <cite data-cite="JANOS">Bergman et.al.
(2020)</cite> proposes the following student enrollment problem. The Admission
Office has data for SAT and GPA scores of the admitted students for the incoming
class, and they would want to offer scholarships to students with the goal of
maximizing the expected number of students that enroll in the college. There is
a total of $n$ students that are admitted. The maximal budget for the sum of all
scholarships offered is $0.2 n \, \text{K\$}$ and each student can be offered a
scholarship of at most $2.5 \, \text{K\$}$.

This problem can be expressed as a mathematical optimization problem as follows.
Two vectors of decision variables $x$ and $y$ of dimension $n$ are used to model
respectively the scholarship offered to each student in $\text{K\$}$ and the
probability that they join. Denoting by $g$ the prediction function for the
probability of the logistic regression we then have for each student $i$:

$$ y_i = g(x_i, SAT_i, GPA_i), $$

with $SAT_i$ and $GPA_i$ the (known) SAT and GPA score of each student.

The objective is to maximize the sum of the $y$ variables and the budget
constraint imposes that the sum of the variables $x$ is less or equal to $0.2n$.
Also, each variable $x_i$ is between 0 and 2.5.

The full model then reads:

$$ \begin{aligned} &\max \sum_{i=1}^n y_i \\
&\text{subject to:}\\
&\sum_{i=1}^n x_i \le 0.2*n,\\
&y_i = g(x_i, SAT_i, GPA_i) & & i = 1, \ldots, n,\\
& 0 \le x \le 2.5. \end{aligned} $$

Note that in this example differently to <cite data-cite="JANOS">Bergman et.al.
(2020)</cite> we scale the features for the regression. Also, to fit in Gurobi's
limited size license we only consider the problem where $n=250$.

We note also that the model may differ from the objectives of Admission Offices
and don't encourage its use in real life. The example is for illustration
purposes only.

## Importing packages and retrieving the data

We import the necessary packages. Besides the usual (`numpy`, `gurobipy`,
`pandas`), for this we will use Scikit-learn's Pipeline, StandardScaler and
LogisticRegression.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gurobipy as gp
from gurobi_ml import add_predictor_constr
```

We now retrieve the historical data used to build the regression from Janos
repository.

The features we use for the regression are `"merit"` (scholarship), `"SAT"` and
`"GPA"` and the target is `"enroll"`. We store those values.

```{code-cell} ipython3
# Base URL for retrieving data
janos_data_url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
historical_data = pd.read_csv(
    janos_data_url + "college_student_enroll-s1-1.csv", index_col=0
)

# classify our features between the ones that are fixed and the ones that will be
# part of the optimization problem
features = ["merit", "SAT", "GPA"]
target = "enroll"
```

## Fit the logistic regression

For the regression, we use a pipeline with a standard scaler and a logistic
regression. We build it using the `make_pipeline` from `scikit-learn`.

```{code-cell} ipython3
# Run our regression
regression = LogisticRegression(random_state=1)
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
pipe.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])
```

### Optimization Model

We now turn to building the mathematical optimization model for Gurobi.

First, retrieve the data for the new students. We won't use all the data there,
we randomly pick 250 students from it.

```{code-cell} ipython3
# Retrieve new data used to build the optimization problem
studentsdata = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)

nstudents = 250

# Select randomly nstudents in the data
studentsdata = studentsdata.sample(nstudents)
```

A non-trivial part of the model is the decision variables that we need for using
Gurobi Machine Learning.

In the mathematical formulation above, we only had two vectors of variables `x`
and `y`. Then each student had associated its score $SAT_i$ and $GPA_i$ that
were fixed parameters in the optimization. For the Gurobi model, we need to
create a matrix of variables that also includes the values of $SAT$ and $GPA$ of
each student. We will fix those variables by giving them the same lower bound
and upper bound.

Therefore, we need to build 2 matrices of variables, one for each set of bounds,
and we need to make sure that they are in the same order as the regression model
would expect.

To do so, we use `pandas` data frames to construct those lower and upper bounds.

To construct the lower bounds, we first make a copy of `studentsdata` and then
add the `"merit"` column with a value of $0$. We then do the same for the upper
bound, except that the value for `"merit"`is $2.5$.

```{code-cell} ipython3
# Construct lower bounds data frame
feat_lb = studentsdata.copy()
feat_lb.loc[:, "merit"] = 0

# Construct upper bounds data frame
feat_ub = studentsdata.copy()
feat_ub.loc[:, "merit"] = 2.5

# Make sure the columns are ordered in the same way as for the regression model.
feat_lb = feat_lb[features]
feat_ub = feat_ub[features]
```

We can now create the variables for our model: `feature_vars` is initialized
using the data frames we just created (be careful that they have to be converted
to `numpy` arrays).

For the rest of the model, we want to recover from the `feature_vars` matrix,
the column corresponding to merit. With `pandas`, we can use the `get_indexer`
function to recover the index of this column in our `MVar` matrix.

```{code-cell} ipython3
# Start with classical part of the model
m = gp.Model()

feature_vars = m.addMVar(
    feat_lb.shape, lb=feat_lb.to_numpy(), ub=feat_ub.to_numpy(), name="feats"
)
y = m.addMVar(nstudents, name="y")

x = feature_vars[:, feat_lb.columns.get_indexer(["merit"])][:, 0]
```

We add the objective and the budget constraint:

```{code-cell} ipython3
m.setObjective(y.sum(), gp.GRB.MAXIMIZE)

m.addConstr(x.sum() <= 0.2 * nstudents)
m.update()
```

Finally, we insert the constraints from the regression. In this model we want to
have use the probability estimate of a student joining the college, so we choose
the parameter `output_type` to be `"probability_1"`. Note that due to the shapes
of the `feature_vars` matrix and `y`, this will insert one regression constraint
for each student.

With the `print_stats` function we display what was added to the model.

```{code-cell} ipython3
pred_constr = add_predictor_constr(
    m, pipe, feature_vars, y, output_type="probability_1"
)

pred_constr.print_stats()
```

We can now optimize the problem.

```{code-cell} ipython3
m.optimize()
```

Remember that for the logistic regression, Gurobi does a piecewise-linear
approximation of the logistic function. We can therefore get some significant
errors when comparing the results of the Gurobi model with what is predicted by
the regression.

We print the error. Here we need to use `get_error_proba`.

```{code-cell} ipython3
print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)
```

The error we get might be considered too large, but we can use Gurobi parameters
to tune the piecewise-linear approximation made by Gurobi (at the expense of a
harder models).

The specific parameters are explained in the documentation of [Functions
Constraints](https://www.gurobi.com/documentation/9.1/refman/constraints.html#subsubsection:GenConstrFunction)
in Gurobi's manual.

We can pass those parameters to the
[add_predictor_constr](../api/AbstractPredictorConstr.rst#gurobi_ml.add_predictor_constr)
function in the form of a dictionary with the keyword parameter
`pwd_attributes`.

Now we want a more precise solution, so we remove the current constraint, add a
new one that does a tighter approximation and resolve the model.

```{code-cell} ipython3
pred_constr.remove()

pwl_attributes = {
    "FuncPieces": -1,
    "FuncPieceLength": 0.01,
    "FuncPieceError": 1e-4,
    "FuncPieceRatio": -1.0,
}
pred_constr = add_predictor_constr(
    m, pipe, feature_vars, y, output_type="probability_1", pwl_attributes=pwl_attributes
)

m.optimize()
```

We can see that the error has been reduced.

```{code-cell} ipython3
print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)
```

+++ {"nbsphinx": "hidden"}

Copyright © 2022 Gurobi Optimization, LLC
