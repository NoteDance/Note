import tensorflow as tf


class max_pool2d:
    def __init__(self,ksize,strides,padding):
        self.ksize=ksize
        self.strides=strides
        self.padding=padding
    
    
    def __call__(self,data):
        return tf.nn.max_pool2d(data,ksize=self.ksize,strides=self.strides,padding=self.padding)