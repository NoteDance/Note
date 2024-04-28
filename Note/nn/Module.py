import tensorflow as tf


class Module:
    param=[]
    param_dict=dict()
    param_dict['dense_weight']=[]
    param_dict['dense_bias']=[]
    param_dict['conv2d_weight']=[]
    param_dict['conv2d_bias']=[]
    layer_dict=dict()
    ctl_list=[]
    ctsl_list=[]
    name=None
    
    
    def cast_param(dict,key,dtype):
        for param in dict[key]:
            param.assign(tf.cast(param,dtype))
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
        Module.ctl_list.clear()
        Module.ctsl_list.clear()
        Module.name=None
        return
