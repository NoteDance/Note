import tensorflow as tf


class dropout:
    def __init__(self,rate,noise_shape=None,seed=None):
        self.rate=rate
        self.noise_shape=noise_shape
        self.seed=seed
        self.train_flag=True
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        if train_flag==True:
            output=tf.nn.dropout(data,self.rate,noise_shape=self.noise_shape,seed=self.seed)
        else:
            output=data
        return output