import tensorflow as tf


def gradient_clipping(grads, method, threshold):
  """
  A function that performs gradient clipping based on the given method and threshold.

  Args:
    grads: a list of tensors, representing the gradients of the parameters
    method: a string, either 'value' or 'norm', indicating the clipping method
    threshold: a scalar, representing the clipping threshold

  Returns:
    clipped_grads: a list of tensors, representing the clipped gradients of the parameters
  """
  if method == 'value':
    # clip the gradients by value
    clipped_grads = [tf.clip_by_value(g, -threshold, threshold) for g in grads]
  elif method == 'norm':
    # clip the gradients by norm
    clipped_grads = [tf.clip_by_norm(g, threshold) for g in grads]
  else:
    # invalid method
    raise ValueError('Invalid clipping method: {}'.format(method))
  
  return clipped_grads


def adaptive_gradient_clipping(grads, params, epsilon=1e-3, max_ratio=1.0):
  """
  A function that performs adaptive gradient clipping based on the ratio of parameter norms to gradient norms.

  Args:
    grads: a list of tensors, representing the gradients of the parameters
    params: a list of tensors, representing the parameters

  Returns:
    clipped_grads: a list of tensors, representing the clipped gradients of the parameters
  """
  # compute the global norm of the parameters
  param_norm = tf.linalg.global_norm(params)
  
  # compute the global norm of the gradients
  grad_norm = tf.linalg.global_norm(grads)
  
  # compute the maximum allowed ratio
  max_ratio = param_norm / tf.maximum(grad_norm, epsilon)
  
  # compute the clipping factor
  clip_factor = tf.minimum(max_ratio, max_ratio)
  
  # clip the gradients by the factor
  clipped_grads = [g * clip_factor for g in grads]
  
  return clipped_grads
