import tensorflow as tf
from Note.nn.layer.zeropadding2d import zeropadding2d


class unfold:
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.zeropadding2d = zeropadding2d(padding=padding)
    
    def __call__(self, data):
        x = self.zeropadding2d(data)
        x = tf.image.extract_patches(x, sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.stride, self.stride, 1], rates=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))
        return x