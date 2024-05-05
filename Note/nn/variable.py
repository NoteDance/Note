import tensorflow as tf
from Note.nn.Module import Module

def variable(initial_value):
    Module.param.append(tf.Variable(initial_value))
    return