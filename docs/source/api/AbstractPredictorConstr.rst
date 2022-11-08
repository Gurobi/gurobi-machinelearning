Generic Predictor Function
==========================

.. autofunction:: gurobi_ml.add_predictor_constr


.. autoclass:: gurobi_ml.modeling.basepredictor.AbstractPredictorConstr
    :members:

    .. method:: get_error()

      Returns error in Gurobi's solution with respect to prediction from input

      Note that this function is implemented in child classes. It is only here for documentation
      purposes.

      :Returns: **error** - Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predict(x)` and
            `y`, where `predict` is the prediction function for the object we are modeling
            (`predict` for Scikit-Learn and Keras, `forward` for Pytorch).

      :Return Type: error: ndarray of same shape as :py:attr:`output`

      :Raises: **NoSolution** - If the Gurobi model has no solution (either was not optimized or is infeasible).

    .. method:: remove()

      Remove from gp_model everything that was added to embed predictor.
