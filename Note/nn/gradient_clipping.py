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