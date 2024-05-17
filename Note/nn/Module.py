import tensorflow as tf


class Module:
    param=[]
    param_dict=dict()
    param_dict['dense_weight']=[]
    param_dict['dense_bias']=[]
    param_dict['conv2d_weight']=[]
    param_dict['conv2d_bias']=[]
    layer_dict=dict()
    layer_param=dict()
    counter=0
    name_list=[]
    ctl_list=[]
    ctsl_list=[]
    name=None
    name_=None
    
    
    def __init__(self):
        Module.init()
        self.param=Module.param
        self.param_dict=Module.param_dict
        self.layer_dict=Module.layer_dict
        self.layer_param=Module.layer_param
    
    
    def add():
        Module.counter+=1
        Module.name_list.append('layer'+str(Module.counter))
        return
    
    
    def apply(func):
        for layer in Module.layer_dict[Module.name]:
            func(layer)
        if len(Module.name_list)>0:
            Module.name_list.pop()
        return
    
    
    def apply_decay(self,str,weight_decay,flag=True):
        if flag==True:
            for param in self.param_dict[str]:
                param.assign(weight_decay * param)
        else:
            for param in self.param_dict[str]:
                param.assign(param / weight_decay)
        return
    
    
    def cast_param(self,key,dtype):
        for param in self.param_dict[key]:
            param.assign(tf.cast(param,dtype))
        return
    
    
    def freeze(self,key):
        for param in self.layer_param[key]:
            param.trainable=False
        return
    
    
    def unfreeze(self,key):
        for param in self.layer_param[key]:
            param.trainable=True
        return
    
    
    def convert_to_list():
        for ctl in Module.ctl_list:
            ctl()
        return
    
    
    def convert_to_shared_list(manager):
        for ctsl in Module.ctsl_list:
            ctsl(manager)
        return
    
    
    def init():
        Module.param.clear()
        Module.param_dict['dense_weight'].clear()
        Module.param_dict['dense_bias'].clear()
        Module.param_dict['conv2d_weight'].clear()
        Module.param_dict['conv2d_bias'].clear()
        Module.layer_dict=dict()
        Module.layer_param=dict()
        Module.counter=0
        Module.name_list=[]
        Module.ctl_list.clear()
        Module.ctsl_list.clear()
        Module.name=None
        Module.name_=None
        return
