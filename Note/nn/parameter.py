import tensorflow as tf
from Note import nn
from Note.nn.Model import Model
import inspect

class Parameter:
    def __new__(cls, data, trainable=True, name=None):
        if name is not None:
            param = tf.Variable(data, trainable=trainable, name=name)
        else:
            param = tf.Variable(data, trainable=trainable)
        
        # Find the layer being initialized by examining the call stack
        frame = inspect.currentframe()
        try:
            # Look up the call stack to find a Layer instance being initialized
            while frame:
                frame = frame.f_back
                if frame and 'self' in frame.f_locals:
                    potential_layer = frame.f_locals['self']
                    if isinstance(potential_layer, nn.Layer):
                        potential_layer.add_param(param)
                        break
        finally:
            del frame
        
        Model.param.append(param)
        if Model.name!=None and Model.name not in Model.layer_param:
            Model.layer_param[Model.name]=[]
            Model.layer_param[Model.name].append(param)
        elif Model.name!=None:
            Model.layer_param[Model.name].append(param)
        
        return param
