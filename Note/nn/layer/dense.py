import tensorflow as tf
from Note import nn
from Note.nn.Model import Model


class dense(nn.Layer): # define a class for dense (fully connected) layer
    def __init__(self,output_size,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,use_bias=True,trainable=True,dtype='float32',name=None): # define the constructor method
        super().__init__()
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=output_size
        self.name=name
        self.init_weights=None
        if input_size!=None:
            if name==None:
                self.weight=nn.initializer([input_size,output_size],weight_initializer,dtype,trainable) # initialize the weight matrix
            else:
                self.weight=nn.initializer([input_size,output_size],weight_initializer,dtype,trainable,name=name)
            Model.param_dict['dense_weight'].append(self.weight)
            if use_bias==True: # if use bias is True
                if name==None:
                    self.bias=nn.initializer([output_size],bias_initializer,dtype,trainable) # initialize the bias vector
                else:
                    self.bias=nn.initializer([output_size],bias_initializer,dtype,trainable,name=name)
                Model.param_dict['dense_bias'].append(self.bias)
            else: # if use bias is False
                self.bias=None # set the bias to None
    
    
    def build(self):
        if self.name==None:
            self.weight=nn.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight matrix
        else:
            self.weight=nn.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype,self.trainable,name=self.name)
        Model.param_dict['dense_weight'].append(self.weight)
        if self.use_bias==True: # if use bias is True
            if self.name==None:
                self.bias=nn.initializer([self.output_size],self.bias_initializer,self.dtype,self.trainable) # initialize the bias vector
            else:
                self.bias=nn.initializer([self.output_size],self.bias_initializer,self.dtype,self.trainable,name=self.name)
            Model.param_dict['dense_bias'].append(self.bias)
        if self.init_weights!=None:
            self.init_weights(self)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        return nn.activation(data,self.weight,self.bias,self.activation,self.use_bias) # return the output of applying activation function to the linear transformation of data and weight, plus bias if use bias is True