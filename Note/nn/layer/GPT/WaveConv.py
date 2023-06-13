import tensorflow as tf
from Note.nn.initializer import i
from Note.nn.activation import activation_dict


'''
The purpose of this layer module is to let the neural network use wave functions 
to process audio data in convolution operations. Wave functions can help the neural
network capture the temporal features of audio data, such as amplitude, frequency, phase, etc. 
This layer module can be used in the following fields:

Speech recognition, such as converting speech signals into text
Speech synthesis, such as converting text into speech signals
Music generation, such as generating music according to style or emotion
Audio classification, such as classifying audio based on content or source
'''
class WaveConv:
    def __init__(self,in_features,out_features,kernel_size,activation=None,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True):
        # initialize the wave convolution layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # kernel_size: the size of the convolution kernel
        # activation: the activation function to apply to the updated features
        # weight_initializer: the method to initialize the weight matrix
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        self.weight=i.initializer([kernel_size,in_features,out_features],weight_initializer,dtype) # initialize the weight matrix with the given initializer and data type
        self.param=[self.weight] # store the weight matrix in a list for later use
        if use_bias:
            self.bias=i.initializer([out_features],bias_initializer,dtype) # initialize the bias vector with zeros and the given data type
            self.param.append(self.bias) # add the bias vector to the weight list
    
    
    def output(self,data,stride=1,padding='same'):
        # define the output function to compute the output features from the input features using wave convolution
        # data: a tensor of shape [batch_size, num_samples, in_features]
        # stride: the stride of the convolution
        # padding: the padding mode of the convolution ('same' or 'valid')
        # return: a tensor of shape [batch_size, num_samples, out_features]
        output=tf.nn.conv1d(data,self.weight,stride,padding) # apply the 1D convolution to the input features using the weight matrix and the given stride and padding mode
        if self.use_bias:
            output=output+self.bias # add the bias vector to the output features
        if self.activation is not None:
            output=self.activation(output) # apply the activation function to the output features
        return output