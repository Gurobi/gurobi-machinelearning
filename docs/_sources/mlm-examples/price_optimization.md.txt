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

# Pricing and Supply problem
*Note: This example is a limited version of the example [How Much Is Too Much? Avocado Pricing and Supply Using Mathematical Optimization](https://github.com/Gurobi/modeling-examples/tree/master/price_optimization). It is only aimed at showing the capabilities of Gurobi Machine Learning.
In particular the model is restricted to a limited number of regions so that it can fits within Gurobi size-limited license. Please visit the full notebook page, for a more detailed explanation about the problem and detailed analysis of the results.*

A [Food Network article](https://www.foodnetwork.com/fn-dish/news/2018/3/avocado-unseats-banana-as-america-s-top-fruit-import-by-value) from March 2017 declared, "Avocado unseats banana as America's top fruit import." This declaration is incomplete and debatable for reasons other than whether  avocado is a fruit. Avocados are expensive.

As a supplier, setting an appropriate avocado price requires a delicate trade-off.
Set it too high and you lose customers. Set it too low and you won't make a profit.
Equipped with good data, the avocado pricing and supply problem is *ripe* with opportunities for demonstrating the power of optimization and data science.

They say when life gives you avocados, make guacamole.
Just like the perfect guacamole needs the right blend of onion, lemon and spices, finding an optimal avocado price needs the right blend of descriptive, predictive and prescriptive analytics.

|<img src="https://github.com/Gurobi/modeling-examples/blob/master/price_optimization/avocado_image_grocery.jpeg?raw=1" width="500" align="center">|
|:--:|
| <b>Avocados: a quintessential corner of a grocery store. Image Credits: [New York Post](https://nypost.com/2022/02/15/us-will-halt-mexico-avocado-imports-as-long-as-necessary/) </b>|


**Goal**: Develop a data science pipeline for pricing and distribution of avocados to maximize revenue.

This notebook walks through a decision-making pipeline that culminates in a mathematical optimization model.
There are three stages:
- First, understand the dataset and infer the relationships between categories such as the sales, price, region, and seasonal trends.
- Second, build a prediction model that predicts the demand for avocados as a function of price, region, year and the seasonality.
- Third, design an optimization problem that sets the optimal price and supply quantity to maximize the net revenue while incorporating costs for wastage and transportation.

## Load the Packages and the Datasets

We use real sales data provided by the [Hass Avocado Board](https://hassavocadoboard.com/) (HAB), whose aim is to "make avocados America’s most popular fruit". This dataset contains consolidated information on several years' worth of market prices and sales of avocados.

We will now load the following packages for analyzing and visualizing the data.


### Extra required packages
- seaborn

```{code-cell}
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import sklearn
import numpy as np
```

The dataset from HAB contains sales data for the years 2019-2022. This data is augmented by a previous download from HAB available on [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020) with sales for the years 2015-2018.

Each row in the dataset is the weekly number of avocados sold and the weekly average price of an avocado categorized by region and type of avocado. There are two types of avocados: conventional and organic. In this notebook, we will only consider the conventional avocados.
There are eight large regions relevant for this example, namely the Great Lakes, Midsouth, North East, Northern New England, South Central, South East, West and Plains.

Now, load the data and store it into a Pandas dataframe.

```{code-cell}
avocado = pd.read_csv('https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/HABdata_2019_2022.csv') # dataset downloaded directly from HAB
avocado_old = pd.read_csv('https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/kaggledata_till2018.csv') # dataset downloaded from Kaggle
avocado = avocado.append(avocado_old, ignore_index=True)
avocado
```

## Prepare the Dataset

We will now prepare the data for making sales predictions. Add new columns to the dataframe for the year and seasonality. Let each year from 2015 through 2022 be given an index from 0 through 7 in the increasing order of the year. We will define the peak season to be the months of February through July. These months are set based on visual inspection of the trends, but you can try setting other months.

```{code-cell}
# Add the index for each year from 2015 through 2022
pd.set_option("display.max_colwidth",10)
avocado['date'] = pd.to_datetime(avocado['date'])
avocado['year'] = pd.DatetimeIndex(avocado['date']).year
avocado['year_index'] = avocado['year'] - 2015
avocado = avocado.sort_values(by='date')

# Define the peak season
avocado['month'] = pd.DatetimeIndex(avocado['date']).month
peak_months = range(2,8)        # <--------- Set the months for the "peak season"
def peak_season(row):
    return 1 if int(row['month']) in peak_months else 0

avocado['peak'] = avocado.apply(lambda row: peak_season(row), axis=1)

# Scale the number of avocados to millions
avocado['units_sold'] = avocado['units_sold']/1000000

# Select only conventional avocados
avocado = avocado[avocado['type'] == 'Conventional']

avocado = avocado[['date','units_sold','price','region','year','month','year_index','peak']].reset_index(drop = True)

avocado
```

## Part I: Observe Trends in the Data

Now, we will infer sales trends in time and seasonality. For simplicity, let's proceed with data from the United States as a whole.

```{code-cell}
df_Total_US = avocado[avocado['region']=='Total_US']
```

### Sales Over the Years

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

mean = df_Total_US.groupby('year')['units_sold'].mean()
std  = df_Total_US.groupby('year')['units_sold'].std()
axes.errorbar(mean.index, mean, xerr=0.5, yerr=2*std, linestyle='')
axes.set_ylabel('Units Sold (millions)')
axes.set_xlabel('Year')

fig.tight_layout()
```

We can see that the sales generally increased over the years, albeit marginally. The dip in 2019 is the effect of the well-documented [2019 avocado shortage](https://abc7news.com/avocado-shortage-season-prices/5389855/) that led to avocados [nearly doubling in price.](https://abc7news.com/avocado-shortage-season-prices/5389855/)

### Seasonality

We will now see the sales trends within a year.

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

mean = df_Total_US.groupby('month')['units_sold'].mean()
std  = df_Total_US.groupby('month')['units_sold'].std()

axes.errorbar(mean.index, mean, xerr=0.5, yerr=2*std, linestyle='')
axes.set_ylabel('Units Sold (millions)')
axes.set_xlabel('Month')

fig.tight_layout()

plt.xlabel('Month')
axes.set_xticks(range(1,13))
plt.ylabel('Units sold (millions)')
plt.show()
```

We see a Super Bowl peak in February and a Cinco de Mayo peak in May.

### Correlations

Now, we will see how the variables are correlated with each other.
The end goal is to predict sales given the price of an avocado, year and seasonality (peak or not).

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
sns.heatmap(df_Total_US[['units_sold', 'price', 'year', 'peak']].corr(),annot=True, center=0,ax=axes)

axes.set_title('Correlations for conventional avocados')
plt.show()
```

As expected, the sales quantity has a negative correlation with the price per avocado. The sales quantity has a positive correlation with the year and season being a peak season.


### Regions

Finally, we will see how the sales differ among the different regions. This will determine the number of avocados that we want to supply to each region.

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

regions = ['Great_Lakes','Midsouth','Northeast','Northern_New_England','SouthCentral','Southeast','West','Plains']
df = avocado[avocado.region.isin(regions)]

mean = df.groupby('region')['units_sold'].mean()
std  = df.groupby('region')['units_sold'].std()

axes.errorbar(range(len(mean)), mean, xerr=0.5, yerr=2*std, linestyle='')

fig.tight_layout()

plt.xlabel('Region')
plt.xticks(range(len(mean)), pd.DataFrame(mean)['units_sold'].index,rotation=20)
plt.ylabel('Units sold (millions)')
plt.show()
```

Clearly, west-coasters love avocados.


## Part II: Predict the Sales

The trends observed in Part I motivate us to construct a prediction model for sales using the independent variables - price, year, region and seasonality.
Henceforth, the sales quantity will be referred to as the *predicted demand*.

Let us now construct a linear regressor for the demand.
Note that the region is a categorical variable.
The linear regressor can be mathematically expressed as:

$$demand = \beta_0 + \beta_1 \cdot price + \sum\limits_{region} \beta^{region}_3 \cdot \mathbb{1}(region)  +  \beta_4 w_{year} \cdot year +  \beta_5  \cdot \mathbb{1}(peak).$$

Here, the $\beta$ values are weights (or "co-efficients") that have to be learned from the data.
Note that the notation $\mathbb{1}(region)$ is an indicator function that takes the value $1$ for each region in the summation. The value of $\mathbb{1}(peak)$ is $1$ if we consider the peak season.

To validate the regression model, we will randomly split the dataset into $80\%$ training and $20\%$ testing data and learn the weights using sklearn.

```{code-cell}
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from gurobi_ml import add_predictor_constr
```

```{code-cell}
regions
```

```{code-cell}
regions = ['Great_Lakes',
           'Midsouth',
           'Northeast',
           'West',
]
```

```{code-cell}
target = 'units_sold'
features = ['price', 'year_index', 'region', 'peak']

y = df.loc[:, target]
X = df.loc[:, features]

subset = X.loc[:, 'region'].isin(regions)
X = X[subset]
y = y[subset]
X
```

```{code-cell}
transform = make_column_transformer((OneHotEncoder(drop="first"), ['region']), remainder='passthrough',
                                    verbose_feature_names_out=False)

X = transform.fit_transform(X)
```

```{code-cell}
# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
```

```{code-cell}
# Train the model
pipeline = make_pipeline(StandardScaler(), MLPRegressor([10]*2))
pipeline.fit(X_train, y_train)

# Get R^2 from test data
y_pred = pipeline.predict(X_test)
print("The R^2 value in the test set is",r2_score(y_test, y_pred))
```

We can observe a good $R^2$ value in the test set. We will now fit the weights to the full dataset.

```{code-cell}
pipeline.fit(X, y)

y_pred_full = pipeline.predict(X)
print("The R^2 value in the full dataset is",r2_score(y, y_pred_full))
```

## Part III: Optimize for Price and Supply of Avocados


Knowing how the price of an avocado affects the demand, how can we set the optimal avocado price?
We don't want to set the price too high, since that could drive demand and sales down. At the same time, setting the price too low could be sub-optimal when maximizing revenue. So what is the sweet spot?

On the distribution logistics, we want to make sure that there are enough avocados across the regions. We can address these considerations in a mathematical optimization model.
An optimization model finds the **best solution** according to an **objective function** such that the solution satisfies a set of **constraints**.
Here, a solution is expressed as a vector of real values or integer values called **decision variables**.
Constraints are a set of equations or inequalities written as a function of the decision variables.

At the start of each week, assume that the total number of available products is finite. This quantity needs to be distributed to the various regions while maximizing net revenue. So there are two key decisions - the price of an avocado in each region, and the number of avocados allocated to each region.

Let us now define some input parameters and notations used for creating the model. The subscript $r$ will be used to denote each region.

### Input Parameters
- $R$: set of regions,
- $d(p,r)$: predicted demand in region $r\in R$ when the avocado per product is $p$,
- $B$: available avocados to be distributed across the regions,
- $c_{waste}$: cost ($\$$) per wasted avocado,
- $c^r_{transport}$: cost ($\$$) of transporting an avocado to region $r \in R$,
- $a^r_{min},a^r_{max}$: minimum and maximum price ($\$$) per avocado for reigon $r \in R$,
- $b^r_{min},b^r_{max}$: minimum and maximum number of avocados allocated to region $r \in R$,

The following code loads the Gurobi python package and initiates the optimization model.
The value of $B$ is set to $30$ million avocados, which is close to the average weekly supply value from the data.
For illustration, let us consider the peak season of 2021.
The cost of wasting an avocado is set to $\$0.10$.
The cost of transporting an avocado ranges between $\$0.10$ to $\$0.50$ based on each region's distance from the southern border, where the [majority of avocado supply comes from](https://www.britannica.com/plant/avocado).
Further, we can set the price of an avocado to not exceed $\$ 2$ apiece.

```{code-cell}
regions = ['Great_Lakes',
 'West',
 'Northeast']
```

```{code-cell}
import gurobipy as gp
from gurobipy import GRB

m = gp.Model("Avocado_Price_Allocation")

# Sets and parameters
R = len(regions)   # set of all regions

B = 15  # total amount ot avocado supply

peak_or_not = 1 # 1 if it is the peak season; 1 if isn't
year = 2022

c_waste = 0.5 # the cost ($) of wasting an avocado

# the cost of transporting an avocado
c_transport = pd.Series({'Great_Lakes': .3,
                         'Midsouth':.1,
                         'Northeast':.4,
                         'Northern_New_England':.5,
                         'SouthCentral':.3,
                         'Southeast':.2,
                         'West':.2,
                         'Plains':.2})
c_transport = c_transport.loc[regions]
# Get the lower and upper bounds from the dataset for the price and the number of products to be stocked
a_min = 0 # minimum avocado price in each region
a_max = 2 # maximum avocado price in each region
b_min = df.groupby('region')['units_sold'].min().loc[regions]  # minimum number of avocados allocated to each region
b_max = df.groupby('region')['units_sold'].max().loc[regions]   # maximum number of avocados allocated to each region
```

```{code-cell}
# Compute bounds for the features

# First create a data frame indexed by the regions and with the features
# as columns
feat_bounds = pd.DataFrame(index=regions, columns=features)
feat_bounds.loc[:, 'price'] = a_min
feat_bounds.loc[:, 'peak'] = peak_or_not
feat_bounds.loc[:, 'year_index'] = year - 2015
for r in regions:
    feat_bounds.loc[r, 'region'] = r

# Now we use our transform to transform the data to the space of the regression
feat_bounds = pd.DataFrame(data=transform.transform(feat_bounds), columns=transform.get_feature_names_out())

# Finally we can store the bounds
# Price is the only feature that is not fixed and has a different upper bound
feat_lb = feat_bounds.copy()
feat_bounds.loc[:, 'price'] = a_max
feat_ub = feat_bounds
```

### Decision Variables

Let us now define the decision variables.
In our model, we want to store the price and number of avocados allocated to each region. We also want variables that track how many avocados are predicted to be sold and how many are predicted to be wasted.
The following notation is used to model these decision variables, indexed for each region $r$.

$p_r$: the price of an avocado ($\$$) in region $r$,

$x_r$: the number of avocados supplied to region $r$,

$s_r = \min \{x_r,d_r(p_r)\}$: the predicted number of avocados sold in region $r$,

$w_r = x_r - s_r$: the predicted number of avocados wasted in region $r$

We will now add the variables to the Gurobi model.

```{code-cell}
x = m.addMVar(R,name="x",lb=b_min,ub=b_max)  # quantity supplied to each region
s = m.addMVar(R,name="s",lb=0)   # predicted amount of sales in each region for the given price
w = m.addMVar(R,name="w",lb=0)   # excess wasteage in each region

# Add variables for the regression
feats = m.addMVar((R, X.shape[1]), lb=feat_lb.to_numpy(), ub=feat_ub.to_numpy(), name='reg_features')
d = m.addMVar((R), lb=-gp.GRB.INFINITY, name='reg_output')

# Get the price variables from the features of the regression
price_index = transform.get_feature_names().index('price')
p = feats[:, price_index]
m.update()
```

### Set the Objective

Next, we will define the objective function: we want to maximize the **net revenue**. The revenue from sales in each region is calculated by the price of an avocado in that region multiplied by the quantity sold there. There are two types of costs incurred: the wastage costs for excess unsold avocados and the cost of transporting the avocados to the different regions.

The net revenue is the sales revenue subtracted by the total costs incurred. We assume that the purchase costs are fixed and are not incorporated in this model.

Using the defined decision variables, the objective can be written as follows.


$$\textrm{maximize} \, \sum_{r}  (p_r \cdot s_r - c_{waste} \cdot w_r - c^r_{transport} \cdot x_r) $$


Let us now add the objective function to the model.

```{code-cell}
m.setObjective(p@s - c_waste * w.sum()- c_transport.to_numpy() @ x)
m.ModelSense = GRB.MAXIMIZE
```

### Add the Supply Constraint

We now introduce the constraints. The first constraint is to make sure that the total number of avocados supplied is equal to $B$, which can be mathematically expressed as follows.

$$\sum_{r} x_r = B$$

The following code adds this constraint to the model.

```{code-cell}
m.addConstr(x.sum() == B)
m.update()
```

### Add Constraints That Define Sales Quantity

Next, we should define the predicted sales quantity in each region.
We can assume that if we supply more than the predicted demand, we sell exactly the predicted demand.
Otherwise, we sell exactly the allocated amount.
Hence, the predicted sales quantity is the minimum of the allocated quantity and the predicted demand, i.e., $s_r = \min \{x_r,d_r(p_r)\}$.
This relationship can be modeled by the following two constraints for each region $r$.

$$
\begin{aligned}
s_r &\leq x_r  \\
s_r &\leq d(p_r,r)
\end{aligned}
$$

These constraints will ensure that the sales quantity $s_r$ in region $r$ is  greater than neither the allocated quantity nor the predicted demand. Note that the maximization objective function tries to maximize the revenue from sales, and therefore the optimizer will maximize the predicted sales quantity. This is assuming that the surplus and transportation costs are less than the sales price per avocado. Hence, these constraints along with the objective will ensure that the sales are equal to the minimum of supply and predicted demand.

Let us now add these constraints to the model.

```{code-cell}
m.addConstr(s <= x)
m.addConstr(s <= d)
m.update()
```

### Add the Wastage Constraints

Finally, we should define the predicted wastage in each region, given by the supplied quantity that is not predicted to be sold. We can express this mathematically for each region $r$.

$$w_r = x_r - s_r$$

We can add these constraints to the model.

```{code-cell}
m.addConstr(w == x - s)
m.update()
```

```{code-cell}
pred_constr = add_predictor_constr(m, pipeline, feats, d)
```

```{code-cell}
pred_constr.print_stats()
```

### Fire Up the Solver

We have added the decision variables, objective function, and the constraints to the model.
The model is ready to be solved.
Before we do so, we should let the solver know what type of model this is.
The default setting assumes that the objective and the constraints are linear functions of the variables.

In our model, the objective is **quadratic** since we take the product of price and the predicted sales, both of which are variables.
Maximizing a quadratic term is said to be **non-convex**, and we specify this by setting the value of the Gurobi parameter **NonConvex** to be $2$.
See [the documentation of the NonConvex parameter](https://www.gurobi.com/documentation/9.5/refman/nonconvex.html) for more details.

```{code-cell}
m.Params.NonConvex = 2
m.optimize()
```

```{code-cell}
print(f"Maximal error in predicted values in solution {np.max(np.abs(pred_constr.get_error()))}")
```

The solver solved the optimization problem in less than a second.
Let us now analyze the optimal solution by storing it in a Pandas dataframe.

```{code-cell}
pd.set_option("display.max_colwidth",50)
solution = pd.DataFrame()
solution['Region'] = regions
solution['Price'] = p.X
solution['Allocated'] = x.X.round(4)
solution['Sold'] = s.X.round(4)
solution['Wasted'] = w.X.round(4)
solution['Pred_demand'] = d.X.round(4)

opt_revenue = m.ObjVal
print("\n The optimal net revenue: $%f million"%opt_revenue)
solution
```

Let us now visualize a scatter plot between the price and the number of avocados sold (in millions) for the eight regions.

```{code-cell}
fig, ax = plt.subplots(1,1)
plot_sol = sns.scatterplot(data=solution,x='Price',y='Sold',hue='Region',s=100)
plot_waste = sns.scatterplot(data=solution,x='Price',y='Wasted',marker='x',hue='Region',s=100,legend = False)

plot_sol.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
plot_waste.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.ylim(0, 10)
plt.xlim(0.5, 2.2)
ax.set_xlabel('Price per avocado ($)')
ax.set_ylabel('Number of avocados sold (millions)')
plt.show()
print("The circles represent sales quantity and the cross markers represent the wasted quantity.")
```

Copyright © 2022 Gurobi Optimization, LLC
