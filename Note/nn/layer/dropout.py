import tensorflow as tf
from Note import nn


class dropout:
    def __init__(self,rate,noise_shape=None,seed=None):
        self.rate=rate
        self.noise_shape=noise_shape
        self.seed=seed
        self.train_flag=True
        nn.Model.layer_list.append(self)
        
    
    def __call__(self,data,train_flag=None):
        if train_flag==None:
            train_flag=self.train_flag
        if train_flag==True:
            output=tf.nn.dropout(data,self.rate,noise_shape=self.noise_shape,seed=self.seed)
        else:
            output=data
        return output
