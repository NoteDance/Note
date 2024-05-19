import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package
from Note.nn.Model import Model


class conv2d: # define a class for 2D convolutional layer
    def __init__(self,filters,kernel_size,input_size=None,strides=[1,1],padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NHWC',dilations=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        if isinstance(kernel_size,int):
            kernel_size=[kernel_size,kernel_size]
        if isinstance(strides,int):
            strides=[strides,strides]
        self.kernel_size=kernel_size
        self.input_size=input_size
        self.strides=strides
        self.padding=padding
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.activation=activation # set the activation function
        self.data_format=data_format
        self.dilations=dilations
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=filters
        if input_size!=None:
            self.weight=i.initializer([kernel_size[0],kernel_size[1],input_size,filters],weight_initializer,dtype) # initialize the weight tensor
            Model.param_dict['conv2d_weight'].append(self.weight)
            if len(Model.name_list)>0:
                Model.name=Model.name_list[-1]
            if Model.name!=None and Model.name not in Model.layer_dict:
                Model.layer_dict[Model.name]=[]
                Model.layer_dict[Model.name].append(self)
            elif Model.name!=None:
                Model.layer_dict[Model.name].append(self)
            if Model.name_!=None and Model.name_ not in Model.layer_param:
                Model.layer_param[Model.name_]=[]
                Model.layer_param[Model.name_].append(self.weight)
            elif Model.name_!=None:
                Model.layer_param[Model.name_].append(self.weight)
            if use_bias==True: # if use bias is True
                self.bias=i.initializer([filters],bias_initializer,dtype) # initialize the bias vector
                Model.param_dict['conv2d_bias'].append(self.bias)
                if Model.name_!=None and Model.name_ not in Model.layer_param:
                    Model.layer_param[Model.name_]=[]
                    Model.layer_param[Model.name_].append(self.bias)
                elif Model.name_!=None:
                    Model.layer_param[Model.name_].append(self.bias)
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            if trainable==False:
                self.param=[]
            Model.param.extend(self.param)
    
    
    def build(self):
        self.weight=i.initializer([self.kernel_size[0],self.kernel_size[1],self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight tensor
        Model.param_dict['conv2d_weight'].append(self.weight)
        if len(Model.name_list)>0:
            Model.name=Model.name_list[-1]
        if Model.name!=None and Model.name not in Model.layer_dict:
            Model.layer_dict[Model.name]=[]
            Model.layer_dict[Model.name].append(self)
        elif Model.name!=None:
            Model.layer_dict[Model.name].append(self)
        if Model.name_!=None and Model.name_ not in Model.layer_param:
            Model.layer_param[Model.name_]=[]
            Model.layer_param[Model.name_].append(self.weight)
        elif Model.name_!=None:
            Model.layer_param[Model.name_].append(self.weight)
        if self.use_bias==True: # if use bias is True
            self.bias=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector
            Model.param_dict['conv2d_bias'].append(self.bias)
            if Model.name_!=None and Model.name_ not in Model.layer_param:
                Model.layer_param[Model.name_]=[]
                Model.layer_param[Model.name_].append(self.bias)
            elif Model.name!=None:
                Model.layer_param[Model.name_].append(self.bias)
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
        if self.trainable==False:
            self.param=[]
        Model.param.extend(self.param)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if self.use_bias==True: # if use bias is True
            return a.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv2d,self.bias) # return the output of applying activation function to the convolution of data and weight, plus bias
        else: # if use bias is False
            return a.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv2d) # return the output of applying activation function to the convolution of data and weight