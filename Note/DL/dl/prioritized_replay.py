import tensorflow as tf
import numpy as np


class pr:
    def __init__(self):
        self.loss=None
        self.index=None
    
    
    def sample(self,train_data,train_labels,alpha,batch):
        prios=(self.loss+1e-7)**alpha
        p=prios/tf.reduce_sum(prios)
        self.index=np.random.choice(np.arange(len(train_data)),size=[batch],p=p.numpy(),replace=False)
        self.batch=batch
        return train_data[self.index],train_labels[self.index]
    
    
    def update(self,loss=None, index=None):
        if loss is not None:
            loss=tf.cast(loss,tf.float32)
            self.loss_.assign(loss)
        elif index is not None:
            self.loss[index[0]:index[1]]=tf.abs(self.loss_[:self.batch])
        else:
            self.loss[self.index]=tf.abs(self.loss_[:self.batch])
        return
