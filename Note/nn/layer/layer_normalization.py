import tensorflow as tf


class layer_normalization:
    def __init__(self,epsilon=1e-6):
        self.epsilon=epsilon
        self.train_flag=True
    
    
    def output(self,data,train_flag=True):
        self.train_flag=train_flag
        if self.train_flag:
            mean,variance=tf.nn.moments(data,axes=[-1],keepdims=True)
            std=tf.math.sqrt(variance)
            normalized=(data-mean)/(std+self.epsilon)
            return normalized
        else:
            return data
