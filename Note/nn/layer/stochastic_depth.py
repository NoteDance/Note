import tensorflow as tf
from Note import nn


class stochastic_depth:
    def __init__(self, drop_path_rate):
        self.drop_path_rate=drop_path_rate
        self.train_flag=True
        nn.Model.layer_list.append(self)
        if nn.Model.name_!=None and nn.Model.name_ not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name_]=[]
            nn.Model.layer_eval[nn.Model.name_].append(self)
        elif nn.Model.name_!=None:
            nn.Model.layer_eval[nn.Model.name_].append(self)
    
    
    def __call__(self, x, train_flag=None):
        if train_flag==None:
            train_flag=self.train_flag
        if train_flag:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
