import tensorflow as tf
from Note.nn.Module import Module

def variable(initial_value):
    variable=tf.Variable(initial_value)
    Module.param.append(variable)
    return variable
