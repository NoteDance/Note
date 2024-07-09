import tensorflow as tf
from Note import nn
from Note.nn.Model import Model


class dense: # define a class for dense (fully connected) layer
    def __init__(self,output_size,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,use_bias=True,trainable=True,dtype='float32',name=None): # define the constructor method
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
        if len(Model.name_list)>0:
            Model.name=Model.name_list[-1]
        if Model.name!=None and Model.name not in Model.layer_dict:
            Model.layer_dict[Model.name]=[]
            Model.layer_dict[Model.name].append(self)
        elif Model.name!=None:
               Model.layer_dict[Model.name].append(self)
        if input_size!=None:
            self.weight=nn.initializer([input_size,output_size],weight_initializer,dtype,trainable) # initialize the weight matrix
            Model.param_dict['dense_weight'].append(self.weight)
            if Model.name_!=None and Model.name_ not in Model.layer_param:
                Model.layer_param[Model.name_]=[]
                Model.layer_param[Model.name_].append(self.weight)
            elif Model.name_!=None:
                Model.layer_param[Model.name_].append(self.weight)
            if use_bias==True: # if use bias is True
                self.bias=nn.initializer([output_size],bias_initializer,dtype,trainable) # initialize the bias vector
                Model.param_dict['dense_bias'].append(self.bias)
                if Model.name_!=None and Model.name_ not in Model.layer_param:
                    Model.layer_param[Model.name_]=[]
                    Model.layer_param[Model.name_].append(self.bias)
                elif Model.name_!=None:
                    Model.layer_param[Model.name_].append(self.bias)
            else: # if use bias is False
                self.bias=None # set the bias to None
            if use_bias==True: # if use bias is True
                if name!=None:
                    self.weight=tf.Variable(self.weight,trainable=trainable,name=name)
                    self.bias=tf.Variable(self.bias,trainable=trainable,name=name)
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                if name!=None:
                    self.weight=tf.Variable(self.weight,trainable=trainable,name=name)
                self.param=[self.weight] # store only the weight in a list
    
    
    def build(self):
        self.weight=nn.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight matrix
        if self.name!=None:
            self.weight=tf.Variable(self.weight,trainable=self.trainable,name=self.name)
        Model.param_dict['dense_weight'].append(self.weight)
        if Model.name_!=None and Model.name_ not in Model.layer_param:
            Model.layer_param[Model.name_]=[]
            Model.layer_param[Model.name_].append(self.weight)
        elif Model.name_!=None:
            Model.layer_param[Model.name_].append(self.weight)
        if self.use_bias==True: # if use bias is True
            self.bias=nn.initializer([self.output_size],self.bias_initializer,self.dtype,self.trainable) # initialize the bias vector
            if self.name!=None:
                self.bias=tf.Variable(self.bias,trainable=self.trainable,name=self.name)
            Model.param_dict['dense_bias'].append(self.bias)
            if Model.name_!=None and Model.name_ not in Model.layer_param:
                Model.layer_param[Model.name_]=[]
                Model.layer_param[Model.name_].append(self.bias)
            elif Model.name_!=None:
                Model.layer_param[Model.name_].append(self.bias)
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
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