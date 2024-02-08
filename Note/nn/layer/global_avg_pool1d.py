import tensorflow as tf


class global_avg_pool1d:
    def __init__(self,keepdims=False):
        self.keepdims=keepdims
    
    
    def __call__(self,data):
        return tf.reduce_mean(data,[1],keepdims=self.keepdims)