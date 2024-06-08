import tensorflow as tf # import the TensorFlow library
from Note import nn
from Note.nn.Model import Model


class conv2d: # define a class for 2D convolutional layer
    def __init__(self,filters,kernel_size,input_size=None,strides=[1,1],padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NHWC',dilations=None,groups=1,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
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
        self.groups=groups
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=filters
        self.weight=[] # initialize an empty list for weight tensors
        self.bias=[] # initialize an empty list for bias vectors
        if not isinstance(padding,str):
            self.zeropadding2d=nn.zeropadding2d(padding=padding)
        if input_size!=None:
            self.groups=groups if input_size%groups==0 else 1
            if self.groups==1:
                self.weight=nn.initializer([kernel_size[0],kernel_size[1],input_size,filters],weight_initializer,dtype,trainable) # initialize the weight tensor
                if use_bias==True: # if use bias is True
                    self.bias=nn.initializer([filters],bias_initializer,dtype,trainable) # initialize the bias vector
                    self.param=[self.weight,self.bias] # store the parameters in a list
                else: # if use bias is False
                    self.param=[self.weight] # store only the weight in a list
            else:
                for i in range(self.groups): # loop over the number of groups
                    self.weight.append(nn.initializer([kernel_size[0],kernel_size[1],input_size//self.groups,filters//self.groups],weight_initializer,dtype,trainable)) # initialize a weight tensor for each group with the given shape, initializer and data type, and append it to the weight list
                    if use_bias==True: # if use bias is True
                        self.bias.append(nn.initializer([filters//self.groups],bias_initializer,dtype,trainable)) # initialize a bias vector for each group with the given shape, initializer and data type, and append it to the bias list
                if use_bias==True: # if use bias is True
                    self.param=self.weight+self.bias # store the parameters in a list by concatenating the weight and bias lists
                else: # if use bias is False
                    self.param=self.weight # store only the weight list as the parameters
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
            if use_bias==True:
                Model.param_dict['conv2d_bias'].append(self.bias)
                if Model.name_!=None and Model.name_ not in Model.layer_param:
                    Model.layer_param[Model.name_]=[]
                    Model.layer_param[Model.name_].append(self.bias)
                elif Model.name_!=None:
                    Model.layer_param[Model.name_].append(self.bias)
            Model.param.extend(self.param)
    
    
    def build(self):
        self.groups=self.groups if self.input_size%self.groups==0 else 1 # check if the number of input channels is divisible by the number of groups, otherwise set it to 1
        if self.groups==1:
            self.weight=nn.initializer([self.kernel_size[0],self.kernel_size[1],self.input_size,self.output_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight tensor
            if self.use_bias==True: # if use bias is True
                self.bias=nn.initializer([self.output_size],self.bias_initializer,self.dtype,self.trainable) # initialize the bias vector
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
        else:
            for i in range(self.groups): # loop over the number of groups
                self.weight.append(nn.initializer([self.kernel_size[0],self.kernel_size[1],self.input_size//self.groups,self.output_size//self.groups],self.weight_initializer,self.dtype,self.trainable)) # initialize a weight tensor for each group with the given shape, initializer and data type, and append it to the weight list
                if self.use_bias==True: # if use bias is True
                    self.bias.append(nn.initializer([self.output_size//self.groups],self.bias_initializer,self.dtype,self.trainable)) # initialize a bias vector for each group with the given shape, initializer and data type, and append it to the bias list
            if self.use_bias==True: # if use bias is True
                self.param=self.weight+self.bias # store the parameters in a list by concatenating the weight and bias lists
            else: # if use bias is False
                self.param=self.weight # store only the weight list as the parameters
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
        if self.use_bias==True:
            Model.param_dict['conv2d_bias'].append(self.bias)
            if Model.name_!=None and Model.name_ not in Model.layer_param:
                Model.layer_param[Model.name_]=[]
                Model.layer_param[Model.name_].append(self.bias)
            elif Model.name!=None:
                Model.layer_param[Model.name_].append(self.bias)
        Model.param.extend(self.param)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if not isinstance(self.padding,str):
            data=self.zeropadding2d(data)
            padding='VALID'
        else:
            padding=self.padding
        if self.groups==1:
            if self.use_bias==True: # if use bias is True
                return nn.activation_conv(data,self.weight,self.activation,self.strides,padding,self.data_format,self.dilations,tf.nn.conv2d,self.bias) # return the output of applying activation function to the convolution of data and weight, plus bias
            else: # if use bias is False
                return nn.activation_conv(data,self.weight,self.activation,self.strides,padding,self.data_format,self.dilations,tf.nn.conv2d) # return the output of applying activation function to the convolution of data and weight
        else:
            input_groups=tf.split(data,self.groups,axis=-1) # split the input data into groups along the last dimension (channel dimension)
            output_groups=[] # initialize an empty list for output groups
            for i in range(self.groups): # loop over the number of groups
                if self.use_bias==True: # if use bias is True
                    output=nn.activation_conv(input_groups[i],self.weight[i],self.activation,self.strides,padding,self.data_format,self.dilations,tf.nn.conv2d,self.bias[i]) # calculate the output of applying activation function to the convolution of input group and weight tensor, plus bias vector
                else: # if use bias is False
                    output=nn.activation_conv(input_groups[i],self.weight[i],self.activation,self.strides,padding,self.data_format,self.dilations,tf.nn.conv2d) # calculate the output of applying activation function to the convolution of input group and weight tensor
                output_groups.append(output) # append the output to the output groups list
            output=tf.concat(output_groups,axis=-1) # concatenate the output groups along the last dimension (channel dimension)
            return output # return the final output
