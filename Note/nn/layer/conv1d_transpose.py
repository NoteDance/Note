import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package


class conv1d_transpose: # define a class for 1D transposed convolutional layer
    def __init__(self,filters,kernel_size,input_size=None,new_steps=None,strides=[1],padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NWC',dilations=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
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
        self.new_steps=new_steps # set the new steps of the output tensor after transposed convolution
        self.output_size=filters
        if input_size!=None:
            self.weight=i.initializer([kernel_size,filters,input_size],weight_initializer,dtype) # initialize the weight tensor
            if use_bias==True: # if use bias is True
                self.bias=i.initializer([filters],bias_initializer,dtype) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            if trainable==False:
                self.param=[]
    
    
    def build(self):
        self.weight=i.initializer([self.kernel_size,self.output_size,self.input_size],self.weight_initializer,self.dtype) # initialize the weight tensor
        if self.use_bias==True: # if use bias is True
            self.bias=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
        if self.trainable==False:
            self.param=[]
        return
    
    
    def output(self,data): # define the output method
        if self.use_bias==True: # if use bias is True
            return a.activation_conv_transpose(data,self.weight,[data.shape[0],self.new_steps,self.output_size],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv1d_transpose,bias=self.bias) # return the output of applying activation function to the transposed convolution of data and weight, plus bias
        else: # if use bias is False
            return a.activation_conv_transpose(data,self.weight,[data.shape[0],self.new_steps,self.output_size],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv1d_transpose) # return the output of applying activation function to the transposed convolution of data and weight
