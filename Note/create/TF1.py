import tensorflow as tf
import numpy as np


class tf1:
    def __init__(self):
        self.accumulator=0
        self.dtype=[]
        self.shape=[]
        self.batch=None
        self.batches=None
        self.index1=None
        self.index2=None
    
    
    def placeholder(self,data=None,name=None):
        self.accumuator+=1
        if data!=None:
            self.dtype.append(data.dtype)
            self.shape.append(data.shape)
        return tf.placeholde(dtype=self.dtype[self.accumuator-1],shape=[None for x in range(len(self.shape[self.accumuator-1]))],name=name)
    
    
    def batch(self,data):
        if self.index1==self.batches*self.batch:
            return np.concatenate([data[self.index1:],data[:self.index2]])
        else:
            return data[self.index1:self.index2]
    
    
    def Gradient(self,train_loss,lr):
        return tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(train_loss)
    
    
    def Momentum(self,train_loss,lr,momentum=0.99):
        return tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum).minimize(train_loss)
    
    
    def RMSprop(self,train_loss,lr):
        return tf.train.RMSPropOptimizer(learning_rate=lr).minimize(train_loss)
    
    
    def Adam(self,train_loss,lr=0.001):
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)