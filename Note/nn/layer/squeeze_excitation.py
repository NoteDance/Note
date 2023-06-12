import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class squeeze_excitation:
    def __init__(self,input_dim,output_dim,ratio,weight_initializer='Xavier',bias_initializer='zeros',activation='relu',dtype='float32',use_bias=True):
        # input_dim: the dimension of the input features
        # output_dim: the dimension of the output features
        # ratio: the reduction ratio for the squeeze operation
        # weight_initializer: the initializer for the weight matrices
        # bias_initializer: the initializer for the bias vectors
        # activation: the activation function for the squeeze operation, default is relu
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.ratio=ratio
        self.activation=activation_dict[activation]
        # initialize the weight matrices and the bias vectors for the squeeze and excitation operations
        self.weight_S=i.initializer([input_dim,output_dim//ratio],weight_initializer,dtype)
        self.bias_S=i.initializer([output_dim//ratio],bias_initializer,dtype)
        self.weight_E=i.initializer([output_dim//ratio,output_dim],weight_initializer,dtype)
        self.bias_E=i.initializer([output_dim],bias_initializer,dtype)
        if use_bias==True:
            self.param_list=[self.weight_S,self.bias_S,self.weight_E,self.bias_E]
        else:
            self.param_list=[self.weight_S,self.weight_E]
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, H, W, C], where N is the number of samples, H is the height, W is the width, and C is the channel number
        # fix the axis parameter to [1,2] for 4D data
        axis=[1,2]
        # compute the global average pooling of the input features
        squeeze_output=tf.reduce_mean(data,axis=axis)
        # compute the squeeze operation and apply activation function
        squeeze_output=tf.matmul(squeeze_output,self.weight_S)
        squeeze_output=tf.nn.bias_add(squeeze_output,self.bias_S)
        squeeze_output=self.activation(squeeze_output)
        # compute the excitation operation and apply sigmoid function
        excitation_output=tf.matmul(squeeze_output,self.weight_E)
        excitation_output=tf.nn.bias_add(excitation_output,self.bias_E)
        excitation_output=tf.nn.sigmoid(excitation_output)
        # reshape the excitation output to match the input shape of [N, 1, 1, C]
        excitation_output=tf.reshape(excitation_output,[-1,1,1,self.output_dim])
        # compute the output as a scaled version of the input features
        output=data*excitation_output
        # return the output data
        return output