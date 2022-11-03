Examples
========

We develop here examples that showcase specific applications where
Gurobi Machine Learning can be used. Each example should be self-contained, and explains
different aspects of using the package.

In :doc:`mlm-examples/2DPeakFunction`, we show how to approximate a nonlinear function with
a neural network to use it in a Gurobi model.

In :doc:`mlm-examples/adversarial_mnist`, we show how to build a model to construct
adversarial examples on the MNIST digit database.

In :doc:`mlm-examples/student_admission`, we use a logistic regression and embed it in
a Gurobi model.

In :doc:`mlm-examples/price_optimization`, we use a linear regression with categorical
features and show how it can be embedded in a Gurobi model [#]_.

All examples can be run as notebooks on mybinder.org

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Gurobi/gurobi-machinelearning/binder?labpath=docs%2Fnotebooks%2Fipynb

.. toctree::
   :maxdepth: 1
   :hidden:

   mlm-examples/2DPeakFunction.md
   mlm-examples/adversarial_mnist.md
   mlm-examples/student_admission.md
   mlm-examples/price_optimization.md

.. rubric:: Footnotes

.. [#] Support for categorical feature is very limited and in particular they can't be
       used easily as variables.
