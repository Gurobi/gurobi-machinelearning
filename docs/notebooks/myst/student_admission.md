---
jupytext:
  formats: ipynb///ipynb,myst///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
import gurobipy as gp

from gurobi_ml import add_predictor_constr

import gurobipy_pandas as gppd
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
scaler = StandardScaler()
regression = LogisticRegression(random_state=1)
pipe = make_pipeline(scaler, regression)
pipe.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])
```

### Optimization Model

We now turn to building the mathematical optimization model for Gurobi.

First, retrieve the data for the new students. We won't use all the data there,
we randomly pick 250 students from it.

```{code-cell} ipython3
# Retrieve new data used to build the optimization problem
studentsdata = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)
```

```{code-cell} ipython3
nstudents = 25

# Select randomly nstudents in the data
studentsdata = studentsdata.sample(nstudents, random_state=1)
```

We can now create the our model.

Since our data is in pandas data frames, we use the package gurobipy-pandas to help create the variables directly using the index of the data frame.

```{code-cell} ipython3
# Start with classical part of the model
m = gp.Model()

# The y variables are modeling the probability of enrollment of each student. They are indexed by students data
y = gppd.add_vars(m, studentsdata, name='enroll_probability')


# We want to complete studentsdata with a column of decision variables to model the "merit" feature.
# Those variable are between 0 and 2.5.
# They are added using the gppd extension and the resulting dataframe is stored in
# students_opt_data.
students_opt_data = studentsdata.gppd.add_vars(m, lb=0.0, ub=2.5, name='merit')

# We denote by x the (variable) "merit" feature
x = students_opt_data.loc[:, "merit"]

# Make sure that studentsdata contains only the features column and in the right order
students_opt_data = students_opt_data.loc[:, features]

m.update()

# Let's look at our features dataframe for the optimization
students_opt_data[:10]
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
of the `studentsdata` data frame and `y`, this will insert one regression constraint
for each student.

With the `print_stats` function we display what was added to the model.

```{code-cell} ipython3
pred_constr = add_predictor_constr(
    m, pipe, students_opt_data, y, output_type="probability_1"
)

pred_constr.print_stats()
```

We can now optimize the problem.
With Gurobi ≥ 11.0, the attribute `FuncNonLinear` is automatically set to 1 by Gurobi machine learning on the nonlinear constraints it adds
in order to deal algorithmically with the logistic function.

Older versions of Gurobi would make a piece-wise linear approximation of the logistic function. You can refer to [older versions
of this documentation](https://gurobi-machinelearning.readthedocs.io/en/v1.3.0/mlm-examples/student_admission.html) for dealing with those approximations.

```{code-cell} ipython3
m.optimize()
```

We print the error using [get_error](../api/AbstractPredictorConstr.rst#gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.get_error) (note that we take the maximal error
over all input vectors).

```{code-cell} ipython3
print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)
```

Finally, note that we can directly get the input values for the regression in a solution as a pandas dataframe using input_values.

```{code-cell} ipython3
pred_constr.input_values
```

+++ {"nbsphinx": "hidden"}

Copyright © 2023 Gurobi Optimization, LLC
