import tensorflow as tf
from Note.nn.Model import Model

def Parameter(data,trainable=True,name=None):
    param=tf.Variable(data,trainable=trainable)
    if name!=None:
        param=tf.Variable(param,name=name)
    Model.param.append(param)
    return param
