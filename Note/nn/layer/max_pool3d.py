import tensorflow as tf


class max_pool3d:
    def __init__(self,ksize,strides,padding):
        self.ksize=ksize
        self.strides=strides
        self.padding=padding
    
    
    def output(self,data):
        return tf.nn.max_pool3d(data,ksize=self.ksize,strides=self.strides,padding=self.padding)