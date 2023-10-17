import tensorflow as tf


class maxout:
    """Applies Maxout to the input.

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville, Yoshua Bengio. https://arxiv.org/abs/1302.4389

    Usually the operation is performed in the filter/channel dimension. This
    can also be used after Dense layers to reduce number of features.

    Args:
      input_shape: Shape of the input tensor.
      num_units: Specifies how many features will remain after maxout
        in the `axis` dimension (usually channel).
        This must be a factor of number of features.
      axis: The dimension where max pooling will be performed. Default is the
        last dimension.

    Input shape:
      nD tensor with shape: `(batch_size, ..., axis_dim, ...)`.

    Output shape:
      nD tensor with shape: `(batch_size, ..., num_units, ...)`.
    """

    def __init__(self, num_units: int, axis: int = -1, input_shape=None):
        self.num_units = num_units
        self.axis = axis
        self.input_shape=input_shape
        if input_shape is not None:
            self.num_channels = self.input_shape[axis]
            if not isinstance(self.num_channels, tf.Tensor) and self.num_channels % self.num_units:
                raise ValueError(
                    "number of features({}) is not "
                    "a multiple of num_units({})".format(self.num_channels, self.num_units)
                )
    
            if axis < 0:
                self.axis_ = axis + len(self.input_shape)
            else:
                self.axis_ = axis
            assert self.axis_ >= 0, "Find invalid axis: {}".format(self.axis)


    def output(self,data):
        if self.input_shape is None:
            self.input_shape=list(data.shape)
            num_channels = self.input_shape[self.axis]
            if not isinstance(num_channels, tf.Tensor) and num_channels % self.num_units:
                raise ValueError(
                    "number of features({}) is not "
                    "a multiple of num_units({})".format(num_channels, self.num_units)
                )
    
            if self.axis < 0:
                axis = self.axis + len(self.input_shape)
            else:
                axis = self.axis
            assert axis >= 0, "Find invalid axis: {}".format(self.axis)
        else:
            axis=self.axis_
            num_channels=self.num_channels
            
        expand_shape = self.input_shape[:]
        expand_shape[axis] = self.num_units
        k = num_channels // self.num_units
        expand_shape.insert(axis, k)

        output = tf.math.reduce_max(
            tf.reshape(data, expand_shape), axis, keepdims=False
        )
        return output
