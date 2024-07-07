import tensorflow as tf
from Note import nn


class avg_pool1d:
    def __init__(self,kernel_size,strides=None,padding=0):
        self.kernel_size=kernel_size
        self.strides=strides if strides!=None else kernel_size
        self.padding=padding
        if not isinstance(padding,str):
            self.zeropadding1d=nn.zeropadding1d(padding=padding)
    
    
    def __call__(self,data):
        if not isinstance(self.padding,str):
            data=self.zeropadding1d(data)
            padding='VALID'
        else:
            padding=self.padding
        return tf.nn.avg_pool1d(data,ksize=self.kernel_size,strides=self.strides,padding=padding)
