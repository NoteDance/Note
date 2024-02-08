import tensorflow as tf


class global_avg_pool2d:
    def __init__(self,keepdims=False):
        self.keepdims=keepdims
    
    
    def __call__(self,data):
        return tf.reduce_mean(data,[1,2],keepdims=self.keepdims)