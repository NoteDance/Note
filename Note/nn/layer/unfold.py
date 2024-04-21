import tensorflow as tf
from Note.nn.layer.zeropadding2d import zeropadding2d


class unfold:
    def __init__(self, kernel, stride=1, padding=0, dilation=1):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.zeropadding2d = zeropadding2d(padding=padding)
    
    def __call__(self, x):
        x = self.zeropadding2d(x)
        x = tf.image.extract_patches(x, sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.stride, self.stride, 1], rates=[1, self.dilation, self.dilation, 1], padding='VALID')
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))
        return x
