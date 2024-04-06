import tensorflow as tf


class Module:
    param=[]
    param_dict=dict()
    param_dict['dense_weight']=[]
    param_dict['dense_bias']=[]
    param_dict['conv2d_weight']=[]
    param_dict['conv2d_bias']=[]
    ctl_list=[]
    ctsl_list=[]
    
    
    def cast_param(key,dtype):
        for param in Module.param_dict[key]:
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
        param_dict=dict()
        param_dict['dense_weight']=[]
        param_dict['dense_bias']=[]
        param_dict['conv2d_weight']=[]
        param_dict['conv2d_bias']=[]
        Module.ctl_list.clear()
        Module.ctsl_list.clear()
        return
