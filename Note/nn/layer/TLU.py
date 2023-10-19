import tensorflow as tf
from Note.nn.initializer import initializer_


class TLU:
    r"""Thresholded Linear Unit.

    An activation function which is similar to ReLU
    but with a learned threshold that benefits models using FRN(Filter Response
    Normalization). Original paper: https://arxiv.org/pdf/1911.09737.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        affine: `bool`. Whether to make it TLU-Affine or not
            which has the form $\max(x, \alpha*x + \tau)$`
    """

    def __init__(
        self,
        input_shape=None,
        affine: bool = False,
        tau_initializer = "zeros",
        alpha_initializer = "zeros",
        dtype='float32'
    ):
        self.affine = affine
        self.tau_initializer = tau_initializer
        if self.affine:
            self.alpha_initializer = alpha_initializer
        self.dtype=dtype
        self.input_shape=input_shape
        if input_shape is not None:
            param_shape = list(input_shape[1:])
            self.tau = initializer_(param_shape, self.tau_initializer, dtype)
            if self.affine:
                self.alpha = initializer_(param_shape, self.alpha_initializer, dtype)


    def output(self, data):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_shape is None:
            self.input_shape=data.shape
            param_shape = list(self.input_shape[1:])
            self.tau = initializer_(param_shape, self.tau_initializer, self.dtype)
            if self.affine:
                self.alpha = initializer_(param_shape, self.alpha_initializer, self.dtype)
        v = self.alpha * data if self.affine else 0
        return tf.maximum(data, self.tau + v)
