import tensorflow as tf
from Note.nn.Model import Model

def variable(initial_value,name=None):
    variable=tf.Variable(initial_value)
    if name!=None:
        variable.name=name
    Model.param.append(variable)
    return variable
