import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package
from Note.nn.Module import Module


class conv2d_transpose: # define a class for 2D transposed convolutional layer
    def __init__(self,filters,kernel_size,input_size=None,new_rc=None,strides=[1,1],padding='VALID',output_padding=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NHWC',dilations=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.kernel_size=kernel_size
        self.input_size=input_size
        self.strides=strides
        self.padding=padding
        self.output_padding=output_padding
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
            self.weight=i.initializer([kernel_size[0],kernel_size[1],filters,input_size],weight_initializer,dtype) # initialize the weight tensor with reversed input and output channels
            if use_bias==True: # if use bias is True
                self.bias=i.initializer([filters],bias_initializer,dtype) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            if trainable==False:
                self.param=[]
            Module.param.extend(self.param)
    
    
    def build(self):
        self.weight=i.initializer([self.kernel_size[0],self.kernel_size[1],self.filters,self.input_size],self.weight_initializer,self.dtype) # initialize the weight tensor with reversed input and output channels
        if self.use_bias==True: # if use bias is True
            self.bias=i.initializer([self.filters],self.bias_initializer,self.dtype) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
        if self.trainable==False:
            self.param=[]
        Module.param.extend(self.param)
        return
    
    
    def output(self,data): # define the output method
        rows = data.shape[1] # get the number of rows in the input data
        cols = data.shape[2] # get the number of columns in the input data
        
        if self.padding == 'SAME': # if padding is 'SAME'
            padding = [tf.math.ceil((self.kernel_size[0] - 1) / 2), tf.math.ceil((self.kernel_size[1] - 1) / 2)] # calculate the padding values for both dimensions as (kernel_size - 1) / 2, rounded up 
        else: # if padding is not 'same'
            padding = [0, 0] # set the padding values to 0 for both dimensions
        
        if self.output_padding == None: # if output_padding is None
            output_padding = [0, 0] # set the output_padding values to 0 for both dimensions
        else: # if output_padding is not None
            output_padding = self.output_padding # use the given output_padding values
        
        new_rows = ((rows - 1) * self.strides[0] + self.kernel_size[0] - 2 * padding[0] + output_padding[0]) # calculate the new number of rows for the output using the formula 
        new_cols = ((cols - 1) * self.strides[1] + self.kernel_size[1] - 2 * padding[1] + output_padding[1]) # calculate the new number of columns for the output using the formula
        
        if self.use_bias==True: # if use bias is True
            return a.activation_conv_transpose(data,self.weight,[data.shape[0],new_rows,new_cols,self.output_size],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv2d_transpose,bias=self.bias) # return the output of applying activation function to the transposed convolution of data and weight, plus bias, using output_shape as output shape
        else: # if use bias is False
            return a.activation_conv_transpose(data,self.weight,[data.shape[0],new_rows,new_cols,self.output_size],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv2d_transpose) # return the output of applying activation function to the transposed convolution of data and weight, using output_shape as output shape
