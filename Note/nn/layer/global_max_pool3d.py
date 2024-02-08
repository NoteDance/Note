import tensorflow as tf


class global_max_pool3d:
    def __init__(self,keepdims=False):
        self.keepdims=keepdims
        
    
    def __call__(self,data):
        return tf.reduce_max(data,[1,2,3],keepdims=self.keepdims)