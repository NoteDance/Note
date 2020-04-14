import tensorflow as tf
import numpy as np


def weight(shape,mean=0,stddev=0.07,name=None):
    return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=tf.float32),name=name)

def bias(shape,mean=0,stddev=0.07,name=None):
    return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=tf.float32),name=name)


