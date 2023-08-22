import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
from Note.nn.initializer import initializer # import the initializer function from Note.nn package


class group_conv2d: # define a class for group convolutional layer
    def __init__(self,weight_shape,num_groups,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True): # define the constructor method
        self.num_groups=num_groups # set the number of groups
        self.weight=[] # initialize an empty list for weight tensors
        self.bias=[] # initialize an empty list for bias vectors
        self.num_groups=num_groups if weight_shape[-2]%num_groups==0 else 1 # check if the number of input channels is divisible by the number of groups, otherwise set it to 1
        for i in range(num_groups): # loop over the number of groups
            self.weight.append(initializer(weight_shape[:-1]+[weight_shape[-1]//num_groups],weight_initializer,dtype)) # initialize a weight tensor for each group with the given shape, initializer and data type, and append it to the weight list
            if use_bias==True: # if use bias is True
                self.bias.append(initializer([weight_shape[-1]//num_groups],bias_initializer,dtype)) # initialize a bias vector for each group with the given shape, initializer and data type, and append it to the bias list
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.output_size=weight_shape[-1]
        if use_bias==True: # if use bias is True
            self.param=self.weight+self.bias # store the parameters in a list by concatenating the weight and bias lists
        else: # if use bias is False
            self.param=self.weight # store only the weight list as the parameters
    
    
    def output(self,data,strides,padding='VALID',data_format='NHWC',dilations=None): # define the output method
        input_groups=tf.split(data,self.num_groups,axis=-1) # split the input data into groups along the last dimension (channel dimension)
        output_groups=[] # initialize an empty list for output groups
        for i in range(self.num_groups): # loop over the number of groups
            if self.use_bias==True: # if use bias is True
                output=a.activation_conv(input_groups[i],self.weight[i],self.activation,strides,padding,data_format,dilations,tf.nn.conv2d,self.bias[i]) # calculate the output of applying activation function to the convolution of input group and weight tensor, plus bias vector
            else: # if use bias is False
                output=a.activation_conv(input_groups[i],self.weight[i],self.activation,strides,padding,data_format,dilations,tf.nn.conv2d) # calculate the output of applying activation function to the convolution of input group and weight tensor
            output_groups.append(output) # append the output to the output groups list
        output=tf.concat(output_groups,axis=-1) # concatenate the output groups along the last dimension (channel dimension)
        return output # return the final output
