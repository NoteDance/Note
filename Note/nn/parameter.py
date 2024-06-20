import tensorflow as tf
from Note.nn.Model import Model

def Parameter(data,trainable=True,name=None):
    param=tf.Variable(data,trainable=trainable)
    if name!=None:
        param=tf.Variable(param,name=name)
    Model.param.append(param)
    if Model.name_!=None and Model.name_ not in Model.layer_param:
        Model.layer_param[Model.name_]=[]
        Model.layer_param[Model.name_].append(param)
    elif Model.name_!=None:
        Model.layer_param[Model.name_].append(param)
    return param
