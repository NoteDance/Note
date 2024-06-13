import tensorflow as tf
from Note import nn


class Model:
    param=[]
    param_dict=dict()
    param_dict['dense_weight']=[]
    param_dict['dense_bias']=[]
    param_dict['conv2d_weight']=[]
    param_dict['conv2d_bias']=[]
    layer_dict=dict()
    layer_param=dict()
    layer_list=[]
    counter=0
    name_list=[]
    ctl_list=[]
    ctsl_list=[]
    name=None
    name_=None
    train_flag=True
    
    
    def __init__(self):
        Model.init()
        self.param=Model.param
        self.param_dict=Model.param_dict
        self.layer_dict=Model.layer_dict
        self.layer_param=Model.layer_param
        self.layer_list=Model.layer_list
        self.head=None
        self.head_=None
        self.ft_flag=0
        self.detach_flag=False
        
    
    def add():
        Model.counter+=1
        Model.name_list.append('layer'+str(Model.counter))
        return
    
    
    def apply(func):
        for layer in Model.layer_dict[Model.name]:
            func(layer)
        if len(Model.name_list)>0:
            Model.name_list.pop()
            if len(Model.name_list)==0:
                Model.name=None
        return
    
    
    def detach(self):
        if self.detach_flag:
            return
        self.param=Model.param.copy()
        self.param_dict=Model.param_dict.copy()
        self.layer_dict=Model.layer_dict.copy()
        self.layer_param=Model.layer_param.copy()
        self.layer_list=Model.layer_list.copy()
        self.detach_flag=True
        return
    
    
    def training(self,flag=False):
        Model.train_flag=flag
        for layer in self.layer_list:
            layer.train_flag=flag
        return
    
    
    def dense(self,num_classes,dim,weight_initializer='Xavier',use_bias=True):
        self.head=nn.dense(num_classes,dim,weight_initializer,use_bias=use_bias)
        return self.head
    
    
    def conv2d(self,num_classes,dim,kernel_size=1,weight_initializer='Xavier',padding='SAME',use_bias=True):
        self.head=nn.conv2d(num_classes,kernel_size,dim,weight_initializer=weight_initializer,padding=padding,use_bias=use_bias)
        return self.head
    
    
    def fine_tuning(self,num_classes,flag=0):
        self.ft_flag=flag
        if flag==0:
            self.head_=self.head
            if isinstance(self.head,nn.dense):
                self.head=nn.dense(num_classes,self.head.input_size,self.head.weight_initializer,use_bias=self.head.use_bias)
            elif isinstance(self.head,nn.conv2d):
                self.head=nn.conv2d(num_classes,self.head.kernel_size,self.head.input_size,weight_initializer=self.head.weight_initializer,padding=self.head.padding,use_bias=self.head.use_bias)
            self.param[-len(self.head.param):]=self.head.param
            for param in self.param[:-len(self.head.param)]:
                param._trainable=False
        elif flag==1:
            for param in self.param[:-len(self.head.param)]:
                param._trainable=True
        else:
            self.head,self.head_=self.head_,self.head
            self.param[-len(self.head.param):]=self.head.param
            for param in self.param[:-len(self.head.param)]:
                param._trainable=True
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
            param._trainable=False
        return
    
    
    def unfreeze(self,key):
        for param in self.layer_param[key]:
            param._trainable=True
        return
    
    
    def convert_to_list():
        for ctl in Model.ctl_list:
            ctl()
        return
    
    
    def convert_to_shared_list(manager):
        for ctsl in Model.ctsl_list:
            ctsl(manager)
        return
    
    
    def init():
        Model.param.clear()
        Model.param_dict['dense_weight'].clear()
        Model.param_dict['dense_bias'].clear()
        Model.param_dict['conv2d_weight'].clear()
        Model.param_dict['conv2d_bias'].clear()
        Model.layer_dict.clear()
        Model.layer_param.clear()
        Model.layer_list.clear()
        Model.counter=0
        Model.name_list=[]
        Model.ctl_list.clear()
        Model.ctsl_list.clear()
        Model.name=None
        Model.name_=None
        Model.train_flag=True
        return
