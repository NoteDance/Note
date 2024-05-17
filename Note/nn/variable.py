import tensorflow as tf
from Note.nn.Module import Module

def variable(initial_value,name=None):
    variable=tf.Variable(initial_value)
    if name!=None:
        variable.name=name
    Module.param.append(variable)
    return variable
