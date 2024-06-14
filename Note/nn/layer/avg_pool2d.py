import tensorflow as tf
from Note import nn


class avg_pool2d:
    def __init__(self,ksize,strides,padding=0):
        self.ksize=ksize
        self.strides=strides
        self.padding=padding
        if not isinstance(padding,str):
            self.zeropadding2d=nn.zeropadding2d(padding=padding)
    
    
    def __call__(self,data):
        if not isinstance(self.padding,str):
            data=self.zeropadding2d(data)
            padding='VALID'
        else:
            padding=self.padding
        return tf.nn.avg_pool2d(data,ksize=self.ksize,strides=self.strides,padding=padding)
