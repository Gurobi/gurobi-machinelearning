---
customTheme: "gurobi"
logoImg: "images/gurobi-logo.png"
title: "ML Constraints in MIP"
author: "Pierre"
description: "Discuss naming and design"
center: "false"
enableTitleFooter: true
slideNumber: "true"
---

<!-- .slide: data-background="images/titlebg.png" class="titlepage" -->
## Machine Learning constraints in Gurobi
### Finding a better title
<br>
<br>
Pierre <br>
Gurobi Optimization

`bonami@gurobi.com`

---

## What it does

- In a Gurobi model we want to insert:
      $$y = g(x)$$
 - Where $g$ is a *trained regressor or classifier*
 - $x$ and $y$ are vectors (matrices) of decision variabes.
 - "$y$ is predicted by $g(x)$"
 - The *trained predictor* could be:
    - neural-net, logistic regression, decision tree,...
    - trained by `scikit-learn`, `pytorch`
    - currently only have regressors

--

## $x$ and $y$ dimensions

- "A trained regressor" is really a function from $\mathbb R^m \rightarrow \mathbb R^n$.
- We should have $x$ and $y$ an $m-$dimenional and $n-$dimensional vectors of variables.
- But also allow them to be matrices of dimensions $k \times m$ and $k \times n$
- Issue: if $n=1$ it's natural to have y a $k-$dimensional vector. Currently allow this.

--

## Example
### Use trained network `nnreg` in model `m`

```python
# minimize function on the input with bounds on output
x = m.addMVar(48)
y = m.addMVar(24, lb=0, ub=10)
# Predict value of y from x using neural network
MLPRegressorPredictor(m, nnreg, x, y)
m.setObjective(x.sum(),GRB.MINIMIZE)
```


---

## Choosing a title

- "Machine Learning constraints in Gurobi"
- "Machine Learning Predictors in Gurobi"
- "Trained Machine Learning Predictors in Gurobi"
- "Using Trained Predictors in Gurobi"
- ...

--

## Things I dislike

- I think "Constraint" is really our terminology and not *datascientist*.
- "Machine Learning" 🔥 but logistic regression or some other predictors could be from computational statistics.
- Not having "Trained" somewhere would make some people believe that we do the training.

---

## Name of the package

Currently named `ml2gurobi`

(inspiration `ps2pdf`)

- `gurobipredictors`
- really not having a good idea

---

## Name of objects we define
### Current naming convention

We take an trained object of class `<class>`.
The class that inserts that object in a Gurobi model will
be `<class><suffix>`:
- Current `<suffix>` is `Predictor`.
- for e.g. `LogisiticRegression` becomes `LogisiticRegressionPredictor`
- Could change the suffix (previously `2Gurobi`, `Constr`, ...), but not too many times (updating all example and tests takes at least an hour)
- Long names for e.g. with `scikit-learn`'s `GradientBoostingRegressor`.
- `Regressor` and `Predictor` are redundant.

--

## Current List of objects that we have

```python
from ml2gurobi.sklearn import (
    LinearRegressionPredictor,
    LogisticRegressionPredictor,
    DecisionTreeRegressorPredictor,
    GradientBoostingRegressorPredictor,
    RandomForestRegressorPredictor,
    MLPRegressorPredictor,
    PipelinePredictor
)
from ml2gurobi.pytorch import SequentialPredictor
```

---

## What we have
### What's missing

- `scikit-learn`: linear regression, logistic regression, neural-network regressor (relu), decision tree, gradient boosting tree, random forest
- `pytorch`: Sequential network of fully connected layers

--

## Missing
- `tensorflow.keras`: most popular deep learning framework
- `ONNX`: Universal file format for deep-learning
- Convolutional layers/Pooling Layers: mostly for image/speech recognition.
- Classification version of tree based estimators
- `XGBoost`: good quality Gradient Boosting trees

--

## Reasonable todo list

- `tensorflow.keras`: similar to `pytorch` actually looks easier to inspect
a trained network: Do
- `ONNX`: -> Imperial College OMLT
- Convolutional Layers/Pooling Layers: mostly a gimmick: don't do.
- Classification version of tree based estimators: Do.
- `XGBoost`: Try to do.
