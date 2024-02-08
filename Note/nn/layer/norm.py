import tensorflow as tf
import numpy as np
from Note.nn.initializer import initializer


class norm:
    """A preprocessing layer which normalizes continuous features.

    This layer will shift and scale inputs into a distribution centered around
    0 with standard deviation 1. It accomplishes this by precomputing the mean
    and variance of the data, and calling `(input - mean) / sqrt(var)` at
    runtime.

    The mean and variance values for the layer must be either supplied on
    construction or learned via `adapt()`. `adapt()` will compute the mean and
    variance of the data and store them as the layer's weights. `adapt()` should
    be called before `fit()`, `evaluate()`, or `predict()`.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        axis: Integer, tuple of integers, or None. The axis or axes that should
          have a separate mean and variance for each index in the shape. For
          example, if shape is `(None, 5)` and `axis=1`, the layer will track 5
          separate mean and variance values for the last axis. If `axis` is set
          to `None`, the layer will normalize all elements in the input by a
          scalar mean and variance. When `-1` the last axis of the
          input is assumed to be a feature dimension and is normalized per
          index. Note that in the specific case of batched scalar inputs where
          the only axis is the batch axis, the default will normalize each index
          in the batch separately. In this case, consider passing `axis=None`.
          Defaults to `-1`.
        mean: The mean value(s) to use during normalization. The passed value(s)
          will be broadcast to the shape of the kept axes above; if the value(s)
          cannot be broadcast, an error will be raised when this layer's
          `build()` method is called.
        variance: The variance value(s) to use during normalization. The passed
          value(s) will be broadcast to the shape of the kept axes above; if the
          value(s) cannot be broadcast, an error will be raised when this
          layer's `build()` method is called.
        invert: If True, this layer will apply the inverse transformation
          to its inputs: it would turn a normalized input back into its
          original form.

    """

    def __init__(self, input_shape=None, axis=-1, mean=None, variance=None, invert=False, dtype='float32'):
        self.input_shape = input_shape
        # Standardize `axis` to a tuple.
        if axis is None:
            axis = ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)
        self.axis = axis

        # Set `mean` and `variance` if passed.
        if isinstance(mean, tf.Variable):
            raise ValueError(
                "Normalization does not support passing a Variable "
                "for the `mean` init arg."
            )
        if isinstance(variance, tf.Variable):
            raise ValueError(
                "Normalization does not support passing a Variable "
                "for the `variance` init arg."
            )
        if (mean is not None) != (variance is not None):
            raise ValueError(
                "When setting values directly, both `mean` and `variance` "
                "must be set. Got mean: {} and variance: {}".format(
                    mean, variance
                )
            )
        self.input_mean = mean
        self.input_variance = variance
        self.invert = invert
        self.dtype = dtype
        if input_shape is not None:
            self.build()
        self.flag=0
    
    
    def build(self, data=None):
        if self.input_shape is None:
            input_shape=data.shape
            self.dtype=data.dtype
        else:
            input_shape=self.input_shape
        
        if isinstance(input_shape, (list, tuple)) and all(
            isinstance(shape, tf.TensorShape) for shape in input_shape
        ):
            raise ValueError(
                "Normalization only accepts a single input. If you are "
                "passing a python list or tuple as a single input, "
                "please convert to a numpy array or `tf.Tensor`."
            )

        input_shape = tf.TensorShape(input_shape).as_list()
        ndim = len(input_shape)

        if any(a < -ndim or a >= ndim for a in self.axis):
            raise ValueError(
                "All `axis` values must be in the range [-ndim, ndim). "
                "Found ndim: `{}`, axis: {}".format(ndim, self.axis)
            )

        # Axes to be kept, replacing negative values with positive equivalents.
        # Sorted to avoid transposing axes.
        self._keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        # All axes to be kept should have known shape.
        for d in self._keep_axis:
            if input_shape[d] is None:
                raise ValueError(
                    "All `axis` values to be kept must have known shape. "
                    "Got axis: {}, "
                    "input shape: {}, with unknown axis at index: {}".format(
                        self.axis, input_shape, d
                    )
                )
        # Axes to be reduced.
        self._reduce_axis = [d for d in range(ndim) if d not in self._keep_axis]
        # 1 if an axis should be reduced, 0 otherwise.
        self._reduce_axis_mask = [
            0 if d in self._keep_axis else 1 for d in range(ndim)
        ]
        # Broadcast any reduced axes.
        self._broadcast_shape = [
            input_shape[d] if d in self._keep_axis else 1 for d in range(ndim)
        ]
        mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)

        if self.input_mean is None:
            self.adapt_mean = initializer(mean_and_var_shape, "zeros", self.dtype)
            self.adapt_variance = initializer(mean_and_var_shape, "ones", self.dtype)
            self.finalize_state()
        else:
            # In the no adapt case, make constant tensors for mean and variance
            # with proper broadcast shape for use during call.
            mean = self.input_mean * np.ones(mean_and_var_shape)
            variance = self.input_variance * np.ones(mean_and_var_shape)
            mean = tf.reshape(mean, self._broadcast_shape)
            variance = tf.reshape(variance, self._broadcast_shape)
            self.mean = tf.cast(mean, self.dtype)
            self.variance = tf.cast(variance, self.dtype)
        return
    
    
    def finalize_state(self):
        if self.input_mean is not None:
            return
    
        # In the adapt case, we make constant tensors for mean and variance with
        # proper broadcast shape and dtype each time `finalize_state` is called.
        self.mean = tf.reshape(self.adapt_mean, self._broadcast_shape)
        self.mean = tf.cast(self.mean, self.dtype)
        self.variance = tf.reshape(self.adapt_variance, self._broadcast_shape)
        self.variance = tf.cast(self.variance, self.dtype)
        return
    
    
    def __call__(self, data):
        if self.input_shape is None and self.flag==0:
            self.flag=1
            self.build(data)
        # The base layer automatically casts floating-point inputs, but we
        # explicitly cast here to also allow integer inputs to be passed
        if self.invert:
            return self.mean + (
                data * tf.maximum(tf.sqrt(self.variance), 1e-07)
            )
        else:
            return (data - self.mean) / tf.maximum(
                tf.sqrt(self.variance), 1e-07
            ) 