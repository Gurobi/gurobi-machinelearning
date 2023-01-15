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

# Price Optimization

<div class="alert alert-info">
Note

This example is adapted from the example in Gurobi's modeling examples [How Much
Is Too Much? Avocado Pricing and Supply Using Mathematical
Optimization](https://github.com/Gurobi/modeling-examples/tree/master/price_optimization).

The main difference is that it uses `Scikit-learn` for the regression model and
Gurobi Machine Learning to embed the regression in a Gurobi model.

But it also differs in that it uses Matrix variables and that the interactive
part of the notebook is skipped. Please refer to the original example for this.

This example illustrates in particular how to use categorical variables in a
regression.

If you are already familiar with the example from the other notebook, you can
jump directly to [building the regression model](#Part-II:-Predict-the-Sales)
and then to [formulating the optimization
problem](#Part-III:-Optimize-for-Price-and-Supply-of-Avocados).
</div>

A [Food Network
article](https://www.foodnetwork.com/fn-dish/news/2018/3/avocado-unseats-banana-as-america-s-top-fruit-import-by-value)
from March 2017 declared, "Avocado unseats banana as America's top fruit
import." This declaration is incomplete and debatable for reasons other than
whether  avocado is a fruit. Avocados are expensive.

As a supplier, setting an appropriate avocado price requires a delicate
trade-off. Set it too high and you lose customers. Set it too low, and you won't
make a profit. Equipped with good data, the avocado pricing and supply problem
is *ripe* with opportunities for demonstrating the power of optimization and
data science.

They say when life gives you avocados, make guacamole. Just like the perfect
guacamole needs the right blend of onion, lemon and spices, finding an optimal
avocado price needs the right blend of descriptive, predictive and prescriptive
analytics.

|<img
src="https://github.com/Gurobi/modeling-examples/blob/master/price_optimization/avocado_image_grocery.jpeg?raw=1"
width="500" align="center">| |:--:| | <b>Avocados: a quintessential corner of a
grocery store. Image Credits: [New York
Post](https://nypost.com/2022/02/15/us-will-halt-mexico-avocado-imports-as-long-as-necessary/)
</b>|


**Goal**: Develop a data science pipeline for pricing and distribution of
avocados to maximize revenue.

This notebook walks through a decision-making pipeline that culminates in a
mathematical optimization model. There are three stages:

- First, understand the dataset and infer the relationships between categories
  such as the sales, price, region, and seasonal trends.
- Second, build a prediction model that predicts the demand for avocados as a
  function of price, region, year and the seasonality.
- Third, design an optimization problem that sets the optimal price and supply
  quantity to maximize the net revenue while incorporating costs for wastage and
  transportation.

+++

## Load the Packages and the Datasets

We use real sales data provided by the [Hass Avocado
Board](https://hassavocadoboard.com/) (HAB), whose aim is to "make avocados
America’s most popular fruit". This dataset contains consolidated information on
several years' worth of market prices and sales of avocados.

We will now load the following packages for analyzing and visualizing the data.

```{code-cell} ipython3
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import sklearn
import numpy as np
```

```{code-cell} ipython3
from gurobi_ml import add_predictor_constr
import gurobipy_pandas as gppd
```

The dataset from HAB contains sales data for the years 2019-2022. This data is
augmented by a previous download from HAB available on
[Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020) with sales
for the years 2015-2018.

Each row in the dataset is the weekly number of avocados sold and the weekly
average price of an avocado categorized by region and type of avocado. There are
two types of avocados: conventional and organic. In this notebook, we will only
consider the conventional avocados. There are eight large regions, namely the
Great Lakes, Midsouth, North East, Northern New England, South Central, South
East, West and Plains.

Now, load the data and store into a Pandas dataframe.

```{code-cell} ipython3
data_url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/"
avocado = pd.read_csv(
    data_url + "HABdata_2019_2022.csv"
)  # dataset downloaded directly from HAB
avocado_old = pd.read_csv(
    data_url + "kaggledata_till2018.csv"
)  # dataset downloaded from Kaggle
avocado = pd.concat([avocado, avocado_old])
avocado
```

## Prepare the Dataset

We will now prepare the data for making sales predictions. Add new columns to
the dataframe for the year and seasonality. Let each year from 2015 through 2022
be given an index from 0 through 7 in the increasing order of the year. We will
define the peak season to be the months of February through July. These months
are set based on visual inspection of the trends, but you can try setting other
months.

```{code-cell} ipython3
# Add the index for each year from 2015 through 2022
avocado["date"] = pd.to_datetime(avocado["date"])
avocado["year"] = pd.DatetimeIndex(avocado["date"]).year
avocado["year_index"] = avocado["year"] - 2015
avocado = avocado.sort_values(by="date")

# Define the peak season
avocado["month"] = pd.DatetimeIndex(avocado["date"]).month
peak_months = range(2, 8)  # <--------- Set the months for the "peak season"


def peak_season(row):
    return 1 if int(row["month"]) in peak_months else 0


avocado["peak"] = avocado.apply(lambda row: peak_season(row), axis=1)

# Scale the number of avocados to millions
avocado["units_sold"] = avocado["units_sold"] / 1000000

# Select only conventional avocados
avocado = avocado[avocado["type"] == "Conventional"]

avocado = avocado[
    ["date", "units_sold", "price", "region", "year", "month", "year_index", "peak"]
].reset_index(drop=True)

avocado
```

## Part 1: Observe Trends in the Data

Now, we will infer sales trends in time and seasonality. For simplicity, let's
proceed with data from the United States as a whole.

```{code-cell} ipython3
df_Total_US = avocado[avocado["region"] == "Total_US"]
```

### Sales Over the Years

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

mean = df_Total_US.groupby("year")["units_sold"].mean()
std = df_Total_US.groupby("year")["units_sold"].std()
axes.errorbar(mean.index, mean, xerr=0.5, yerr=2 * std, linestyle="")
axes.set_ylabel("Units Sold (millions)")
axes.set_xlabel("Year")

fig.tight_layout()
```

We can see that the sales generally increased over the years, albeit marginally.
The dip in 2019 is the effect of the well-documented [2019 avocado
shortage](https://abc7news.com/avocado-shortage-season-prices/5389855/) that led
to avocados [nearly doubling in
price.](https://abc7news.com/avocado-shortage-season-prices/5389855/)

+++

### Seasonality

We will now see the sales trends within a year.

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

mean = df_Total_US.groupby("month")["units_sold"].mean()
std = df_Total_US.groupby("month")["units_sold"].std()

axes.errorbar(mean.index, mean, xerr=0.5, yerr=2 * std, linestyle="")
axes.set_ylabel("Units Sold (millions)")
axes.set_xlabel("Month")

fig.tight_layout()

plt.xlabel("Month")
axes.set_xticks(range(1, 13))
plt.ylabel("Units sold (millions)")
plt.show()
```

We see a Super Bowl peak in February and a Cinco de Mayo peak in May.

+++

### Correlations

Now, we will see how the variables are correlated with each other. The end goal
is to predict sales given the price of an avocado, year and seasonality (peak or
not).

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
sns.heatmap(
    df_Total_US[["units_sold", "price", "year", "peak"]].corr(),
    annot=True,
    center=0,
    ax=axes,
)

axes.set_title("Correlations for conventional avocados")
plt.show()
```

As expected, the sales quantity has a negative correlation with the price per
avocado. The sales quantity has a positive correlation with the year and season
being a peak season.

+++

### Regions

Finally, we will see how the sales differ among the different regions. This will
determine the number of avocados that we want to supply to each region.

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

regions = [
    "Great_Lakes",
    "Midsouth",
    "Northeast",
    "Northern_New_England",
    "SouthCentral",
    "Southeast",
    "West",
    "Plains",
]
df = avocado[avocado.region.isin(regions)]

mean = df.groupby("region")["units_sold"].mean()
std = df.groupby("region")["units_sold"].std()

axes.errorbar(range(len(mean)), mean, xerr=0.5, yerr=2 * std, linestyle="")

fig.tight_layout()

plt.xlabel("Region")
plt.xticks(range(len(mean)), pd.DataFrame(mean)["units_sold"].index, rotation=20)
plt.ylabel("Units sold (millions)")
plt.show()
```

Clearly, west-coasters love avocados.

+++

## Part II: Predict the Sales

The trends observed in Part I motivate us to construct a prediction model for
sales using the independent variables- price, year, region and seasonality.
Henceforth, the sales quantity will be referred to as the *predicted demand*.

Let us now construct a linear regressor for the demand. Note that the region is
a categorical variable, to encode it for the regression we will use the
`OneHotEncoder` of `Scikit-learn`.

Because Gurobi Machine Learning doesn't support this column transformation at
this point, we need to apply the transform to the data directly **before**
applying the regression. We will then show below how to use it to build the
model.

+++

We prepare the data using `OneHotEncoder` and `make_column_transformer`. We want
to transform the region feature using the encoder while the other features
should be unchanged.

Furthermore, we store in X the transformed data and in y the target value
units_sold.

```{code-cell} ipython3
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

feat_transform = make_column_transformer(
    (OneHotEncoder(drop="first"), ["region"]),
    (StandardScaler(), ["price", "year_index"]),
    ("passthrough", ["peak"]),
    verbose_feature_names_out=False,
    remainder='drop'
)

X = df[["region", "price", "year_index", "peak"]]
y = df["units_sold"]
```

To validate the regression model, we will randomly split the dataset into $80\%$
training and $20\%$ testing data and learn the weights using `Scikit-learn`.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1
)
```

Finally, create the regression model and train it.

```{code-cell} ipython3
time()
```

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone
from time import time
regressions = {"Linear Regression": {"regressor":LinearRegression()},
               "MLP Regression": {"regressor": MLPRegressor([10*2])},
               "Decision Tree": {"regressor": DecisionTreeRegressor()},
               "Random Forest": {"regressor": RandomForestRegressor()},
               "Gradient Boosting":
               {"regressor" : GradientBoostingRegressor()}}
```

```{code-cell} ipython3
for regression, data in regressions.items():
    lin_reg = make_pipeline(feat_transform,
                            data["regressor"])
    train_start = time()
    lin_reg.fit(X_train, y_train)
    data["train_time"] = time() - train_start
    data["pipeline"] = lin_reg

    # Get R^2 from test data
    y_pred = lin_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    data["R2"] = r2
    print(f"The R^2 value in the test set is {r2}")
```

```{code-cell} ipython3
regressions_poly = {}
for regression, data in regressions.items():
    data = {"regressor": clone(data["regressor"])}
    lin_reg = make_pipeline(feat_transform, PolynomialFeatures(),
                            data["regressor"])
    train_start = time()
    lin_reg.fit(X_train, y_train)
    data["train_time"] = time() - train_start
    data["pipeline"] = lin_reg

    # Get R^2 from test data
    y_pred = lin_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    data["R2"] = r2
    regressions_poly[f"{regression} with polynomial features"] = data

    print(f"The R^2 value in the test set is {r2}")
```

```{code-cell} ipython3
regressions |= regressions_poly
```

We can observe a good $R^2$ value in the test set. We will now train the fit the
weights to the full dataset.

```{code-cell} ipython3
lin_reg.fit(X, y)

y_pred_full = lin_reg.predict(X)
print(f"The R^2 value in the full dataset is {r2_score(y, y_pred_full)}")
```

## Part III: Optimize for Price and Supply of Avocados

```{code-cell} ipython3
import gurobipy as gp
from gurobipy import GRB

# Sets and parameters
R = len(regions)  # set of all regions

B = 30  # total amount ot avocado supply

peak_or_not = 1  # 1 if it is the peak season; 1 if isn't
year = 2020

c_waste = 0.1  # the cost ($) of wasting an avocado
# the cost of transporting an avocado
c_transport = pd.Series(
    {
        "Great_Lakes": 0.3,
        "Midsouth": 0.1,
        "Northeast": 0.4,
        "Northern_New_England": 0.5,
        "SouthCentral": 0.3,
        "Southeast": 0.2,
        "West": 0.2,
        "Plains": 0.2,
    }, name='transport_cost'
)

c_transport = c_transport.loc[regions]
# the cost of transporting an avocado

# Get the lower and upper bounds from the dataset for the price and the number of products to be stocked
a_min = 0  # minimum avocado price in each region
a_max = 2  # maximum avocado price in each region

data = pd.concat([c_transport,
                  df.groupby("region")["units_sold"].min().rename('min_delivery'),
                  df.groupby("region")["units_sold"].max().rename('max_delivery')], axis=1)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
m = gp.Model("Avocado_Price_Allocation")

x = gppd.add_vars(m, data, name="x", lb='min_delivery', ub='max_delivery')
s = gppd.add_vars(m, data, name="s") # predicted amount of sales in each region for the given price).
w = gppd.add_vars(m, data, name="w") # excess wasteage in each region).
d = gppd.add_vars(m, data, lb=-gp.GRB.INFINITY, name="demand") # Add variables for the regression
p = gppd.add_vars(m, data, name="price", lb=a_min, ub=a_max)
m.update()

m.setObjective((p * s).sum() - c_waste * w.sum() - (c_transport * x).sum())
m.ModelSense = GRB.MAXIMIZE

m.addConstr(x.sum() == B)
m.update()

gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, x)
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, d)
m.update()

gppd.add_constrs(m, w, gp.GRB.EQUAL, x - s)
m.update()
```

```{code-cell} ipython3
feats = pd.DataFrame(
    data={
        "year_index": year - 2015,
        "peak": peak_or_not,
        "region": regions,
    },
    index=regions
)
feats = pd.concat([feats, p], axis=1)[["region", "price", "year_index", "peak"]]
```

```{code-cell} ipython3
for regression, data in regressions.items():
    pred_constr = add_predictor_constr(m, data["pipeline"], feats, d)

    pred_constr.print_stats()

    m.Params.NonConvex = 2
    m.write(f"{regression}.rlp")
    try:
        start = time()
        m.optimize()
        data["opt_time"] = time() - start
    except:
        data["opt_time"] = float('nan')
        break
        pass
    pred_constr.remove()
```

```{code-cell} ipython3
files = !ls -l *.rlp

sizes = {line[46:-4]: line.split()[4] for line in files}
```

```{code-cell} ipython3
for key, value in sizes.items():
    regressions[key]["file_size"] = value
```

```{code-cell} ipython3
regressions["Linear Regression"]
```

```{code-cell} ipython3
pd.DataFrame.from_dict(regressions, orient='index').drop(["regressor", "pipeline"], axis=1)
```

The solver solved the optimization problem in less than a second. Let us now
analyze the optimal solution by storing it in a Pandas dataframe.

```{code-cell} ipython3
solution = pd.DataFrame(index=regions)

solution["Price"] = p.gppd.X
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = w.gppd.X
solution["Pred_demand"] = d.gppd.X

opt_revenue = m.ObjVal
print("\n The optimal net revenue: $%f million" % opt_revenue)
solution.round(4)
```

We can also check the error in the estimate of the Gurobi solution for the regression model.

```{code-cell} ipython3
print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)
```

And the computed features of the regression model in a pandas dataframe.

```{code-cell} ipython3
pred_constr.input_values
```

Let us now visualize a scatter plot between the price and the number of avocados
sold (in millions) for the eight regions.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1)

plot_sol = sns.scatterplot(data=solution, x="Price", y="Sold", hue=solution.index, s=100)
plot_waste = sns.scatterplot(
    data=solution, x="Price", y="Wasted", marker="x", hue=solution.index, s=100, legend=False
)

plot_sol.legend(loc="center left", bbox_to_anchor=(1.25, 0.5), ncol=1)
plot_waste.legend(loc="center left", bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.ylim(0, 5)
plt.xlim(1, 2.2)
ax.set_xlabel("Price per avocado ($)")
ax.set_ylabel("Number of avocados sold (millions)")
plt.show()
print(
    "The circles represent sales quantity and the cross markers represent the wasted quantity."
)
```

We have shown how to model the price and supply optimization problem with Gurobi
Machine Learning. In the [Gurobi modeling examples
notebook](https://github.com/Gurobi/modeling-examples/tree/master/price_optimization)
more analysis of the solutions this model can give is done interactively. Be
sure to take look at it.

+++ {"nbsphinx": "hidden"}

Copyright © 2022 Gurobi Optimization, LLC
