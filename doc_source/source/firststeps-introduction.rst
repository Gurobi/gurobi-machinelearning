Basics
======

Introduction
------------

The integration of Machine Learning (ML) techniques and Mathematical
Optimization (MO) is a topic of growing interest. One particular approach is to
be able to use *trained* ML models in optimization models
(:cite:t:`JANOS`, :cite:t:`Maragano.et.al2021`, :cite:t:`ceccon2022omlt`). In this approach, the
features and the prediction of the ML model become decision variables of the
optimization while its parameters are fixed. We say that the ML model is
embedded in the optimization model, and we refer to the features and predictions
as input and output variables respectively. They are linked in that, in a
feasible solution, the output variables values are the values predicted by the
regression from the input variables values.

Gurobi Machine Learning is an open-source Python package to embed *trained*
regression models [#]_ in a `gurobipy model
<https://www.gurobi.com/documentation/current/refman/py_model.html>`_ to be
solved with the `Gurobi <https://www.gurobi.com>`_ solver.

The aim of the package is to:

   #. Provide a simple way to embed regression models in a gurobipy model.
   #. Promote a deeper integration between predictive and prescriptive
      analytics.
   #. Allow to easily compare the effect of using different regression models in
      the same mathematical optimization application.
   #. Improve the algorithmic performance of Gurobi on those models.

The package currently supports various `scikit-learn
<https://scikit-learn.org/stable/>`_ objects. It has limited support for the
`Keras <https://keras.io/>`_ API of `TensorFlow <https://www.tensorflow.org/>`_
and `PyTorch <https://pytorch.org/>`_. Only neural networks with ReLU activation
can be used with these two packages.

The package is actively developed and users are encouraged to :doc:`contact us
<meta-contactus>` if they have applications where they use a regression model
that is currently not available.

Below, we give basic installation and usage instructions.

Install
-------

We encourage to install the package via pip (or add it to your
`requirements.txt` file):


.. code-block:: console

  (.venv) pip install gurobi-machinelearning


.. note::

  If not already installed, this should install the ``gurobipy`` and ``numpy``
  packages.


.. note::

  The package has been tested with and is supported for Python 3.9 and Python
  3.10.

  Table table-versions_ lists the version of the relevant packages that are
  tested and supported in the current version (|version|).

  .. _table-versions:

  .. list-table:: Supported packages with version |version|
     :widths: 50 50
     :align: center
     :header-rows: 1

     * - Package
       - Version
     * - ``gurobipy``
       - |GurobiVersion|
     * - ``numpy``
       - |NumpyVersion|
     * - ``pandas``
       - |PandasVersion|
     * - ``torch``
       - |TorchVersion|
     * - ``scikit-learn``
       - |SklearnVersion|
     * - ``tensorflow``
       - |TensorflowVersion|

  Installing any of the machine learning packages is only required if the
  predictor you want to insert uses them (i.e. to insert a Keras based predictor
  you need to have ``tensorflow`` installed).


Usage
-----

The main function provided by the package is
:py:func:`gurobi_ml.add_predictor_constr`. It takes as arguments: a `gurobipy
model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_, a
:doc:`supported regression model <mlm-supported>`, input `Gurobi variables
<https://www.gurobi.com/documentation/current/refman/variables.html>`_ and
output `Gurobi variables
<https://www.gurobi.com/documentation/current/refman/variables.html>`_.

By invoking the function, the gurobipy model is augmented with variables and
constraints so that, in a solution, the values of the output variables are
predicted by the regression model from the values of the input variables. More
formally, if we denote by :math:`g` the prediction function of the regression
model, by :math:`x` the input variables and by :math:`y` the output variables,
:math:`y = g(x)` in any solution.

The function :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>`
returns a modeling object derived from the class
:py:class:`AbstractPredictorConstr
<gurobi_ml.modeling.AbstractPredictorConstr>`. That object keeps track of all
the variables and constraints that have been added to the gurobipy model to
establish the relationship between input and output variables of the regression.

The modeling object can perform a few tasks:

   * Everything it created (i.e. variables and constraints to establish the
     relationship between input and output) can be removed with the
     :py:meth:`remove <gurobi_ml.modeling.AbstractPredictorConstr.remove>`
     member function.
   * It can print a summary of what it added with the :py:meth:`print_stats
     <gurobi_ml.modeling.AbstractPredictorConstr.print_stats>` member function.
   * Once Gurobi computed a solution to the optimization problem, it can compute
     the difference between what the regression model predicts from the input
     values and the values of the output variables in Gurobi's solution with the
     :py:meth:`get_error
     <gurobi_ml.modeling.AbstractPredictorConstr.print_stats>` member function.


The function :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` is
a shorthand that should add the correct model for any supported regression
model, but individual functions for each regression model are also available.
For the list of frameworks and regression models supported and the corresponding
functions please refer to the :doc:`mlm-supported` section.

For some regression models, additional optional parameters can be set to tune
the way the predictor is inserted in the Gurobi model. Those are documented in
the corresponding function linked from :doc:`mlm-supported`.

For a simple example on how to use the package please refer to
:doc:`firststeps-simple-example`. More advanced examples are available
in the :doc:`mlm-examples` section.

License
-------

Gurobi Machine Learning is distributed under the Apache License 2.0.

Note that Gurobi itself is a commercial software and requires a license. When
installed via pip or conda, gurobipy ships with a `limited license
<https://pypi.org/project/gurobipy/>`_ which can only solve models of limited
size. All the examples given in this documentation can be solved using
this license.

If you are a student or staff member of an academic institution you qualify for
a free, full product license of Gurobi. For more information, see:

    https://www.gurobi.com/academia/academic-program-and-licenses/

For a commercial evaluation, you can request an `evaluation license
<https://www.gurobi.com/free-trial/?utm_source=internal&utm_medium=documentation&utm_campaign=fy21_pipinstall_eval_pypipointer&utm_content=c_na&utm_term=pypi>`_.

.. rubric:: Footnotes

.. [#] Classification models are currently not supported (except binary logistic
    regression) but it is planned to add support to some models over time.
