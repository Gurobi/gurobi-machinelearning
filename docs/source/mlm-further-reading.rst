Further Reading
###############

The approach of embedding machine learning models in optimization models has
received sustained attention in recent years with several publications and
published research software packages. Here, we make an informal annotated
bibliography of the works in which we have been interested when developing the
package. We don't claim to be exhaustive.

The JANOS framework was proposed in :cite:t:`JANOS` with an associated `Python
package <https://github.com/INFORMSJoC/2020.1023>`_. The package works with
various Scikit-learn models and solves optimization problems with Gurobi. The
:doc:`mlm-examples/student_admission` example was proposed in that paper.

Another framework is reluMIP, :cite:t:`reluMIP.2021`. It is mostly
aimed at neural networks with ReLU activation formulated with TensorFlow. The
same authors study in particular the use of neural networks in surrogate models, e.g.
:cite:t:`GRIMSTAD2019106580`.

The OptiCL framework was proposed in :cite:t:`Maragano.et.al2021`. An associated
python package, :cite:t:`OptiCL`, is available. The package can
model several Scikit-learn objects. The authors proposed several
interesting applications: palatable diet, cancer treatment. They also propose
original algorithmic approaches to ensure credible predictions and avoid extrapolations.

Finally, among research software packages, OMLT (:cite:t:`ceccon2022omlt`) is a
Python package that supports a variety of neural network structures (dense
layers, convolutional layers, pooling layers) and gradient boosting trees. It is
hooked with the `ONNX <https://onnx.ai/>`_ open format. It is actively developed
and evolving. It is in particular aimed at studying alternative formulations for
the neural network structures.

There is a growing literature on efficient MIP formulation for neural networks.
:cite:t:`Strong-mixed-integer-programming-formulations-for-trained-FULL`,
:cite:t:`The-Convex-Relaxation-Barrier-Revisited` and :cite:t:`betweensteps` are
good starting points.
