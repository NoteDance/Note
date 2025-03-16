import tensorflow as tf
from Note.nn.Model import Model

def Parameter(data,trainable=True,name=None):
    param=tf.Variable(data,trainable=trainable)
    if name!=None:
        param=tf.Variable(param,name=name)
    Model.param.append(param)
    if Model.name!=None and Model.name not in Model.layer_param:
        Model.layer_param[Model.name]=[]
        Model.layer_param[Model.name].append(param)
    elif Model.name_!=None:
        Model.layer_param[Model.name].append(param)
    return param
