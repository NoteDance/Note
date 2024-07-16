import tensorflow as tf
from Note import nn
from typing import Union


class lp_pool1d:
    r"""Apply a 1D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    """
    def __init__(self, norm_type: Union[int, float], kernel_size, strides = None):
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.strides = strides
        if strides is not None:
            self.avg_pool1d = nn.avg_pool1d(kernel_size, strides, 0)
        else:
            self.avg_pool1d = nn.avg_pool1d(kernel_size, padding=0)
    
    def __call__(self, input):
        if self.strides is not None:
            out = self.avg_pool1d(tf.pow(input, self.norm_type))
        else:
            out = self.avg_pool1d(tf.pow(input, self.norm_type))
    
        return tf.pow((tf.sign(out) * tf.nn.relu(tf.abs(out))) * self.kernel_size, (1.0 / self.norm_type))