import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class bottleneck:
    def __init__(self,input_dim,output_dim,bottleneck_dim,weight_initializer='Xavier',bias_initializer='zeros',activation='relu',dtype='float32',use_bias=True):
        # input_dim: the dimension of the input features
        # output_dim: the dimension of the output features
        # bottleneck_dim: the dimension of the bottleneck features
        # weight_initializer: the initializer for the weight matrices
        # bias_initializer: the initializer for the bias vectors
        # activation: the activation function for the bottleneck layer, default is relu
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.bottleneck_dim=bottleneck_dim
        self.activation=activation_dict[activation]
        # initialize the weight matrices and the bias vectors for the bottleneck layer
        self.weight_1=i.initializer([input_dim,bottleneck_dim],weight_initializer,dtype)
        self.bias_1=i.initializer([bottleneck_dim],bias_initializer,dtype)
        self.weight_2=i.initializer([bottleneck_dim,output_dim],weight_initializer,dtype)
        self.bias_2=i.initializer([output_dim],bias_initializer,dtype)
        if use_bias==True:
            self.param_list=[self.weight_1,self.bias_1,self.weight_2,self.bias_2]
        else:
            self.param_list=[self.weight_1,self.weight_2]
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, input_dim], where N is the number of samples
        # compute the first linear transformation and apply activation function
        bottleneck_output=tf.matmul(data,self.weight_1)
        bottleneck_output=tf.nn.bias_add(bottleneck_output,self.bias_1)
        bottleneck_output=self.activation(bottleneck_output)
        # compute the second linear transformation and apply activation function
        output=tf.matmul(bottleneck_output,self.weight_2)
        output=tf.nn.bias_add(output,self.bias_2)
        return output