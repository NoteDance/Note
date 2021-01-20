import tensorflow as tf
import numpy as np


class tf1:
    def __init__(self):
        self.accumulator=0
        self.test_accumulator=0
        self.dtype=[]
        self.shape=[]
    
    
    def placeholder(self,data=None,name=None,test=False):
        if test==False:
            self.accumulator+=1
        else:
            self.test_accumulator+=1
        if data!=None and test==False:
            self.dtype.append(data.dtype)
            self.shape.append(data.shape)
        if test==False:
            return tf.placeholde(dtype=self.dtype[self.accumulator-1],shape=[None for x in range(len(self.shape[self.accumulator-1]))],name=name)
        else:
            return tf.placeholde(dtype=self.dtype[self.test_accumulator-1],shape=[None for x in range(len(self.shape[self.test_accumulator-1]))],name=name)


def Gradient(train_loss,lr):
    return tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(train_loss)


def Momentum(train_loss,lr,momentum=0.99):
    return tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum).minimize(train_loss)


def RMSprop(train_loss,lr):
    return tf.train.RMSPropOptimizer(learning_rate=lr).minimize(train_loss)


def Adam(train_loss,lr=0.001):
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)




def extend(parameter):
    for i in range(len(parameter)-1):
        parameter[0].extend(parameter[i+1])
    return parameter[0]


def apply_gradient(tape,optimizer,loss,parameter):
    gradient=tape.gradient(loss,parameter)
    optimizer.apply_gradients(zip(gradient,parameter))
    return


def einsun(data,weight):
    return tf.einsum('ijk,kl->ijl',data,weight)


def norm(data):
    mean=tf.reduce_mean(data,axis=-1)
    var=(data-tf.expand_dims(mean,axis=-1))**2
    data=(data-mean)/tf.sqrt(var+1e-07)
    return data
