import tensorflow as tf


class Module:
    param=[]
    param_table=dict()
    param_table['dense_weight']=[]
    param_table['dense_bias']=[]
    param_table['conv2d_weight']=[]
    param_table['conv2d_bias']=[]
    ctl_list=[]
    ctsl_list=[]
    
    
    def cast_param(key,dtype):
        for param in Module.param_table[key]:
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
        param_table=dict()
        param_table['dense_weight']=[]
        param_table['dense_bias']=[]
        param_table['conv2d_weight']=[]
        param_table['conv2d_bias']=[]
        Module.ctl_list.clear()
        Module.ctsl_list.clear()
        return
