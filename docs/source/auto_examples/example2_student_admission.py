r"""
Student Enrollment
==================

In this example, we show how to reproduce the model of student
enrollment from Bergman et.al. (2020) with Gurobi Machine Learning.

This model was developed in the context of the development of
`Janos <https://github.com/INFORMSJoC/2020.1023>`__, a toolkit similar
to Gurobi Machine Learning to integrate ML models and Mathematical
Optimization.

This example illustrates in particular how to use the logistic
regression.

We also show how to deal with fixed features in the optimization model
using pandas data frames.

In this model, data of students admissions in a college is used to
predict the probability that a student enrolls to the college.

The data has 3 features: the SAT and GPA scores of each student, and the
scholarship (or merit) that was offered to each student. Finally, it is
known if each student decided to join the college or not.

Based on this data a logistic regression is trained to predict the
probability that a student joins the college.

Using this regression model, Bergman et.al. (2020) proposes the
following student enrollment problem. The Admission Office has data for
SAT and GPA scores of the admitted students for the incoming class, and
they would want to offer scholarships to students with the goal of
maximizing the expected number of students that enroll in the college.
There is a total of :math:`n` students that are admitted. The maximal
budget for the sum of all scholarships offered is
:math:`0.2 n \, \text{K\$}` and each student can be offered a
scholarship of at most :math:`2.5 \, \text{K\$}`.

This problem can be expressed as a mathematical optimization problem as
follows. Two vectors of decision variables :math:`x` and :math:`y` of
dimension :math:`n` are used to model respectively the scholarship
offered to each student in :math:`\text{K\$}` and the probability that
they join. Denoting by :math:`g` the prediction function for the
probability of the logistic regression we then have for each student
:math:`i`:

.. math::  y_i = g(x_i, SAT_i, GPA_i),

with :math:`SAT_i` and :math:`GPA_i` the (known) SAT and GPA score of
each student.

The objective is to maximize the sum of the :math:`y` variables and the
budget constraint imposes that the sum of the variables :math:`x` is
less or equal to :math:`0.2n`. Also, each variable :math:`x_i` is
between 0 and 2.5.

The full model then reads:

.. math::

    \begin{aligned} &\max \sum_{i=1}^n y_i \\
   &\text{subject to:}\\
   &\sum_{i=1}^n x_i \le 0.2*n,\\
   &y_i = g(x_i, SAT_i, GPA_i) & & i = 1, \ldots, n,\\
   & 0 \le x \le 2.5. \end{aligned}

Note that in this example differently to Bergman et.al. (2020) we scale
the features for the regression. Also, to fit in Gurobi’s limited size
license we only consider the problem where :math:`n=250`.

We note also that the model may differ from the objectives of Admission
Offices and don’t encourage its use in real life. The example is for
illustration purposes only.

Importing packages and retrieving the data
------------------------------------------

We import the necessary packages. Besides the usual (``numpy``,
``gurobipy``, ``pandas``), for this we will use Scikit-learn’s Pipeline,
StandardScaler and LogisticRegression.

"""

import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gurobi_ml import add_predictor_constr

######################################################################
# We now retrieve the historical data used to build the regression from
# Janos repository.
#
# The features we use for the regression are ``"merit"`` (scholarship),
# ``"SAT"`` and ``"GPA"`` and the target is ``"enroll"``. We store those
# values.
#

# Base URL for retrieving data
janos_data_url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
historical_data = pd.read_csv(
    janos_data_url + "college_student_enroll-s1-1.csv", index_col=0
)

# classify our features between the ones that are fixed and the ones that will be
# part of the optimization problem
features = ["merit", "SAT", "GPA"]
target = "enroll"


######################################################################
# Fit the logistic regression
# ---------------------------
#
# For the regression, we use a pipeline with a standard scaler and a
# logistic regression. We build it using the ``make_pipeline`` from
# ``scikit-learn``.
#

# Run our regression
scaler = StandardScaler()
regression = LogisticRegression(random_state=1)
pipe = make_pipeline(scaler, regression)
pipe.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])


######################################################################
# Optimization Model
# ~~~~~~~~~~~~~~~~~~
#
# We now turn to building the mathematical optimization model for Gurobi.
#
# First, retrieve the data for the new students. We won’t use all the data
# there, we randomly pick 250 students from it.
#

# Retrieve new data used to build the optimization problem
studentsdata = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)

nstudents = 25

# Select randomly nstudents in the data
studentsdata = studentsdata.sample(nstudents, random_state=1)


######################################################################
# We can now create the our model.
#
# Since our data is in pandas data frames, we use the package
# gurobipy-pandas to help create the variables directly using the index of
# the data frame.
#

# Start with classical part of the model
m = gp.Model()

# The y variables are modeling the probability of enrollment of each student. They are indexed by students data
y = gppd.add_vars(m, studentsdata, name="enroll_probability")


# We want to complete studentsdata with a column of decision variables to model the "merit" feature.
# Those variable are between 0 and 2.5.
# They are added using the gppd extension and the resulting dataframe is stored in
# students_opt_data.
students_opt_data = studentsdata.gppd.add_vars(m, lb=0.0, ub=2.5, name="merit")

# We denote by x the (variable) "merit" feature
x = students_opt_data.loc[:, "merit"]

# Make sure that studentsdata contains only the features column and in the right order
students_opt_data = students_opt_data.loc[:, features]

m.update()

# Let's look at our features dataframe for the optimization
students_opt_data[:10]


######################################################################
# We add the objective and the budget constraint:
#

m.setObjective(y.sum(), gp.GRB.MAXIMIZE)

m.addConstr(x.sum() <= 0.2 * nstudents)
m.update()


######################################################################
# Finally, we insert the constraints from the regression. In this model we
# want to have use the probability estimate of a student joining the
# college, so we choose the parameter ``output_type`` to be
# ``"probability_1"``. Note that due to the shapes of the ``studentsdata``
# data frame and ``y``, this will insert one regression constraint for
# each student.
#
# With the ``print_stats`` function we display what was added to the
# model.
#

pred_constr = add_predictor_constr(
    m, pipe, students_opt_data, y, output_type="probability_1"
)

pred_constr.print_stats()


######################################################################
# We can now optimize the problem. With Gurobi ≥ 11.0, the attribute
# ``FuncNonLinear`` is automatically set to 1 by Gurobi machine learning
# on the nonlinear constraints it adds in order to deal algorithmically
# with the logistic function.
#
# Older versions of Gurobi would make a piece-wise linear approximation of
# the logistic function. You can refer to `older versions of this
# documentation <https://gurobi-machinelearning.readthedocs.io/en/v1.3.0/mlm-examples/student_admission.html>`__
# for dealing with those approximations.
#

m.optimize()


######################################################################
# We print the error using
# :func:`get_error<gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.get_error>`
# (note that we take the maximal error over all input vectors).
#

print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)


######################################################################
# Finally, note that we can directly get the input values for the
# regression in a solution as a pandas dataframe using input_values.
#

pred_constr.input_values


######################################################################
# Copyright © 2023-2026 Gurobi Optimization, LLC
#
