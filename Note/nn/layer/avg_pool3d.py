import tensorflow as tf
from Note import nn


class avg_pool3d:
    def __init__(self,kernel_size,strides,padding=0):
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        if not isinstance(padding,str):
            self.zeropadding3d=nn.zeropadding3d(padding=padding)
    
    
    def __call__(self,data):
        if not isinstance(self.padding,str):
            data=self.zeropadding3d(data)
            padding='VALID'
        else:
            padding=self.padding
        return tf.nn.avg_pool3d(data,ksize=self.kernel_size,strides=self.strides,padding=padding)
