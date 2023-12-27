import tensorflow as tf
from Note.nn.initializer import initializer_
from collections import abc
from Note.nn.Module import Module

class RMSNorm:
  def __init__(
      self,
      axis,
      eps = 1e-5,
      scale_init = None,
      create_scale = True,
      param_axis = None,
      ):
    if not create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if isinstance(axis, slice):
      self.axis = axis
    elif isinstance(axis, int):
      self.axis = (axis,)
    elif (isinstance(axis, abc.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      self.axis = tuple(axis)
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self.eps = eps
    self.create_scale = create_scale
    self.scale_init = scale_init or tf.ones
    if param_axis is None:
      self.param_axis = (-1,)
    else:
      self.param_axis = to_axes_or_slice(param_axis)

  def output(self, data):
    """Connects the layer norm.

    Args:
      inputs: An array, where the data format is ``[N, ..., C]``.

    Returns:
      The normalized array, of the same shape as the inputs.
    """
    axis = self.axis
    if isinstance(axis, slice):
      axis = tuple(range(data.ndim)[axis])

    param_axis = to_abs_axes(self.param_axis, data.ndim)
    if param_axis == (data.ndim - 1,):
      # For param_axis=-1 we store non-broadcast param shape for compatibility
      # with older checkpoints.
      param_shape = data.shape[-1:]
    else:
      param_shape = tuple(
          (data.shape[i] if i in param_axis else 1)
          for i in range(data.ndim))
    if self.create_scale:
      if type(self.scale_init)==str or list:
          scale = initializer_(param_shape, self.scale_init, data.dtype)
      else:
          scale = self.scale_init(param_shape, data.dtype)
          Module.param.append(scale)
      scale = tf.broadcast_to(scale, data.shape)
    else:
      scale = 1.

    mean_squared = tf.math.reduce_mean(tf.math.square(data), axis=axis, keepdims=True)
    mean_squared = tf.broadcast_to(mean_squared, data.shape)

    return data * scale * tf.math.rsqrt(mean_squared + self.eps)

def to_axes_or_slice(axis):
  if isinstance(axis, slice):
    return axis
  elif isinstance(axis, int):
    return (axis,)
  elif (isinstance(axis, abc.Iterable) and
        all(isinstance(ax, int) for ax in axis)):
    return tuple(axis)
  else:
    raise ValueError(
        f"`axis` should be an int, slice or iterable of ints. Got: {axis}")

def to_abs_axes(axis, ndim):
  if isinstance(axis, slice):
    return tuple(range(ndim)[axis])
  else:
    return tuple(sorted({a % ndim for a in axis}))