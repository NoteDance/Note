import tensorflow as tf
from Note.nn.initializer import i
from Note.nn.activation import activation_dict


'''
The purpose of this layer module is to let the neural network use cycle structures 
to process circular data in convolution operations. Cycle structures can help the 
neural network capture the periodicity and the continuity of the data. 
This layer module can be used in the following fields:

Time series analysis, such as forecasting or anomaly detection
Signal processing, such as filtering or compression
Music generation, such as creating melodies or rhythms
Graph neural networks, such as learning cycle representations or cycle classification
'''
class CycleConv:
    def __init__(self,in_features,out_features,kernel_size,activation=None,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True):
        # initialize the cycle convolution layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # kernel_size: the size of the convolution kernel
        # activation: the activation function to apply to the updated features
        # weight_initializer: the method to initialize the weight matrix
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.kernel_size=kernel_size
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        self.weight=i.initializer([kernel_size,in_features,out_features],weight_initializer,dtype) # initialize the weight matrix with the given initializer and data type
        self.weight_list=[self.weight] # store the weight matrix in a list for later use
        if use_bias:
            self.bias=i.initializer([out_features],bias_initializer,dtype) # initialize the bias vector with zeros and the given data type
            self.weight_list.append(self.bias) # add the bias vector to the weight list
    
    
    def output(self,data,stride=1,padding='same'):
        # define the output function to compute the output features from the input features using cycle convolution
        # data: a tensor of shape [batch_size, num_nodes, in_features]
        # stride: the stride of the convolution
        # padding: the padding mode of the convolution ('same' or 'valid')
        # return: a tensor of shape [batch_size, num_nodes, out_features]
        output=tf.concat([data,data[:,:self.kernel_size-1,:]],axis=1) # concatenate the first k-1 nodes to the end of the data to form a cycle
        output=tf.nn.conv1d(output,self.weight,self.stride,self.padding) # apply the 1D convolution to the input features using the weight matrix and the given stride and padding mode
        if self.use_bias:
            output=output+self.bias # add the bias vector to the output features
        if self.activation is not None:
            output=self.activation(output) # apply the activation function to the output features
        return output
