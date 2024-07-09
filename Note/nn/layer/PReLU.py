import tensorflow as tf
from Note.nn.initializer import initializer


class PReLU:
    """Parametric Rectified Linear Unit.

    It follows:

    ```
        f(x) = alpha * x for x < 0
        f(x) = x for x >= 0
    ```

    where `alpha` is a learned array with the same shape as x.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        alpha_initializer: Initializer function for the weights.
        shared_axes: The axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    """

    def __init__(
        self,
        input_shape=None,
        alpha_initializer="zeros",
        shared_axes=None,
        dtype='float32'
    ):
        self.alpha_initializer = alpha_initializer
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        self.dtype=dtype
        self.input_shape=input_shape
        if input_shape is not None:
            param_shape = list(input_shape[1:])
            if self.shared_axes is not None:
                for i in self.shared_axes:
                    param_shape[i - 1] = 1
            self.alpha = initializer(
                shape=param_shape,
                initializer=alpha_initializer,
                dtype=dtype
            )
            self.param=[self.alpha]


    def __call__(self, data):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_shape is None:
            self.input_shape=data.shape
            param_shape = list(self.input_shape[1:])
            if self.shared_axes is not None:
                for i in self.shared_axes:
                    param_shape[i - 1] = 1
            self.alpha = initializer(
                shape=param_shape,
                initializer=self.alpha_initializer,
                dtype=self.dtype
            )
            self.param=[self.alpha]
        pos = tf.nn.relu(data)
        neg = -self.alpha * tf.nn.relu(-data)
        return pos + neg