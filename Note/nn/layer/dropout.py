import tensorflow as tf
from Note import nn


class dropout:
    def __init__(self,rate,noise_shape=None,seed=None):
        self.rate=rate
        self.noise_shape=noise_shape
        self.seed=seed
        self.train_flag=True
        nn.Model.layer_list.append(self)
        if nn.Model.name!=None and nn.Model.name not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name]=[]
            nn.Model.layer_eval[nn.Model.name].append(self)
        elif nn.Model.name!=None:
            nn.Model.layer_eval[nn.Model.name].append(self)
        
    
    def __call__(self,data,training=None):
        if training==None:
            training=self.train_flag
        if training==True:
            output=tf.nn.dropout(data,self.rate,noise_shape=self.noise_shape,seed=self.seed)
        else:
            output=data
        return output
