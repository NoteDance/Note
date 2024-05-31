import tensorflow as tf # import the TensorFlow library
from Note import nn
from Note.nn.initializer import initializer # import the initializer module from Note.nn package
from Note.nn.Model import Model


class group_conv1d: # define a class for 1D convolutional layer
    def __init__(self,filters,kernel_size,num_groups=1,input_size=None,strides=[1],padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NWC',dilations=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        if isinstance(kernel_size,int):
            kernel_size=kernel_size
        elif len(kernel_size)==1:
            kernel_size=kernel_size[0]
        if isinstance(strides,int):
            strides=strides
        elif len(strides)==1:
            strides=strides[0]
        self.kernel_size=kernel_size
        self.num_groups=num_groups
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
        self.weight=[]
        self.bias=[]
        if input_size!=None:
            if len(Model.name_list)>0:
                Model.name=Model.name_list[-1]
            if Model.name!=None and Model.name not in Model.layer_dict:
                Model.layer_dict[Model.name]=[]
                Model.layer_dict[Model.name].append(self)
            elif Model.name!=None:
                Model.layer_dict[Model.name].append(self)
            self.num_groups=num_groups if input_size%num_groups==0 else 1
            for i in range(self.num_groups):
                self.weight.append(initializer([kernel_size,input_size//self.num_groups,filters//self.num_groups],weight_initializer,dtype,trainable)) # initialize the weight tensor
                if use_bias==True: # if use bias is True
                    self.bias.append(initializer([filters//self.num_groups],bias_initializer,dtype,trainable)) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=self.weight+self.bias # store the parameters in a list
            else: # if use bias is False
                self.param=self.weight # store only the weight in a list
            Model.param.extend(self.param)
    
    
    def build(self):
        if len(Model.name_list)>0:
            Model.name=Model.name_list[-1]
        if Model.name!=None and Model.name not in Model.layer_dict:
            Model.layer_dict[Model.name]=[]
            Model.layer_dict[Model.name].append(self)
        elif Model.name!=None:
            Model.layer_dict[Model.name].append(self)
        self.num_groups=self.num_groups if self.input_size%self.num_groups==0 else 1
        for i in range(self.num_groups):
            self.weight.append(initializer([self.kernel_size,self.input_size//self.num_groups,self.output_size//self.num_groups],self.weight_initializer,self.dtype,self.trainable)) # initialize the weight tensor
            if self.use_bias==True: # if use bias is True
                self.bias.append(initializer([self.output_size//self.num_groups],self.bias_initializer,self.dtype,self.trainable)) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=self.weight+self.bias # store the parameters in a list
        else: # if use bias is False
            self.param=self.weight # store only the weight in a list
        Model.param.extend(self.param)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        input_groups=tf.split(data,self.num_groups,axis=-1)
        output_groups=[]
        for i in range(self.num_groups):
            if self.use_bias==True: # if use bias is True
                output=nn.activation_conv(input_groups[i],self.weight[i],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv1d,self.bias[i]) # return the output of applying activation function to the convolution of data and weight, plus bias
            else: # if use bias is False
                output=nn.activation_conv(input_groups[i],self.weight[i],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv1d) # return the output of applying activation function to the convolution of data and weight
            output_groups.append(output)
        output=tf.concat(output_groups,axis=-1)
        return output