### Adapted from TF repo

import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.
  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.
  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.
  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.
  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.
  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.
  Raises:
    ValueError: `xs` and `v` have different length.
  """ 

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = gradients(ys, xs)

  # grads = xs

  assert len(grads) == length

  elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
  ]

  # Second backprop  
  grads_with_none = gradients(elemwise_products, xs)
  return_grads = [
      grad_elem if grad_elem is not None \
      else tf.zeros_like(x) \
      for x, grad_elem in zip(xs, grads_with_none)]
  
  return return_grads


def _AsList(x):
  return x if isinstance(x, (list, tuple)) else [x]

def hessians(ys, xs, name="hessians", colocate_gradients_with_ops=False, 
            gate_gradients=False, aggregation_method=None):
  """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.
  `hessians()` adds ops to the graph to output the Hessian matrix of `ys` 
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` 
  where each tensor is the Hessian of `sum(ys)`. This function currently
  only supports evaluating the Hessian with respect to (a list of) one-
  dimensional tensors.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
  Returns:
    A list of Hessian matrices of `sum(y)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid or not supported. Currently,
      this function only supports one-dimensional `x` in `xs`.
  """
  xs = _AsList(xs)
  kwargs = {
      'colocate_gradients_with_ops': colocate_gradients_with_ops,
      'gate_gradients': gate_gradients,
      'aggregation_method': aggregation_method
    }
  # Compute a hessian matrix for each x in xs
  hessians = []
  for i, x in enumerate(xs):
    # Check dimensions
    ndims = x.get_shape().ndims
    if ndims is None:
      raise ValueError('Cannot compute Hessian because the dimensionality of '
                       'element number %d of `xs` cannot be determined' % i)
    elif ndims != 1:
      raise ValueError('Computing hessians is currently only supported for '
                       'one-dimensional tensors. Element number %d of `xs` has '
                       '%d dimensions.' % (i, ndims))
    with ops.name_scope(name + '_first_derivative'):
      # Compute the partial derivatives of the input with respect to all 
      # elements of `x`
      _gradients = tf.gradients(ys, x, **kwargs)[0]
      # Unpack the gradients into a list so we can take derivatives with 
      # respect to each element
      _gradients = array_ops.unpack(_gradients)
    with ops.name_scope(name + '_second_derivative'):
      # Compute the partial derivatives with respect to each element of the list
      _hess = [tf.gradients(_gradient, x, **kwargs)[0] for _gradient in _gradients]
      # Pack the list into a matrix and add to the list of hessians
      hessians.append(array_ops.pack(_hess, name=name))
  return hessians