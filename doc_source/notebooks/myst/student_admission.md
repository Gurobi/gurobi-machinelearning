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

# Integrate a logistic regression in a Gurobi model
*Note: The resulting model in this example will be too large for a size-limited license; in order to solve it, please visit https://www.gurobi.com/free-trial for a full license*

We take the model from JANOS example:

$
\begin{align}
&\max \sum y_i \\
&\text{subject to:}\\
&\sum x_i \le 100,\\
&y_i = g(x_i, \psi),\\
& 0 \le x \le 2.5.
\end{align}
$

Where, $\psi$ is a matrix of known features. And $g$ is a logistic function computed using the  logistic regression of scikit-learn.

Note that differently to JANOS, we scale the feature corresponding to $x$ for the regression.

```{code-cell}
import gurobipy as gp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gurobi_ml import add_predictor_constr

# Base URL for retrieving data
janos_data_url = 'https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/'
```

```{code-cell}
historical_data = pd.read_csv(janos_data_url + 'college_student_enroll-s1-1.csv', index_col=0)

# classify our features between the ones that are fixed and the ones that will be
# part of the optimization problem
known_features = ["SAT", "GPA"]
dec_features = ["merit"]
target = "enroll"
features = known_features + dec_features

historical_data = historical_data[features + [target]]
```

### Do our logistic regression

```{code-cell}
# Run our regression
X = historical_data.loc[:, features]
Y = historical_data.loc[:, target]
scaler = StandardScaler()
regression = LogisticRegression(random_state=1, penalty='l1', C=10, solver='saga')
pipe = make_pipeline(scaler, regression)
pipe.fit(X=X, y=Y)
```

### Now start with the optimization model

- Read in our data
- add the x and y variables and the regular matrix constraints

```{code-cell}
# Retrieve new data used to build the optimization problem
studentsdata = pd.read_csv(janos_data_url + 'college_applications6000.csv', index_col=0)

studentsdata = studentsdata[known_features]
nstudents = 500

# Select randomly nstudents in the data
studentsdata = studentsdata.sample(nstudents)
```

```{code-cell}
# Start with classical part of the model
m = gp.Model()

knownidx = historical_data.columns.get_indexer(known_features)
scholarshipidx = historical_data.columns.get_indexer(dec_features)

lb = np.zeros((nstudents, len(features)))
ub = np.ones((nstudents, len(features))) * gp.GRB.INFINITY
lb[:, knownidx] = studentsdata.loc[:, known_features]
ub[:, knownidx] = studentsdata.loc[:, known_features]

x = m.addMVar(lb.shape, lb=lb, ub=ub, name="x")
scholarship = x[:, scholarshipidx][:, 0]
y = m.addMVar(nstudents, ub=1, name="y")

scholarship.LB = 0.0
scholarship.UB = 2.5

m.setObjective(y.sum(), gp.GRB.MAXIMIZE)
m.addConstr(scholarship.sum() <= 0.2 * nstudents)

pred_constr = add_predictor_constr(m, pipe, x, y)
```

```{code-cell}
pred_constr.print_stats()
```

### Finally optimize it

```{code-cell}
m.optimize()
```

### Look at solution

```{code-cell}
# This is what we predicted
plt.scatter(scholarship.X, y.X)
```

```{code-cell}
# This is the historical data
plt.scatter(X.loc[:, "merit"], pipe.predict_proba(X)[:, 1])
```

```{code-cell}
# Proportion of students offered a scholarship
print(
    "In historical data {:.4}% students offered a scholarship".format(
        100 * (sum(historical_data.loc[:, "merit"] > 0) / len(historical_data.loc[:, "merit"]))
    )
)
print("In our solution {:.4}% students offered a scholarship".format(100 * sum(scholarship.X > 0) / nstudents))
```

Copyright © 2022 Gurobi Optimization, LLC
