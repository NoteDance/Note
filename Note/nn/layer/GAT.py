import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class GAT:
    def __init__(self,input_dim,output_dim,num_heads,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True):
        # input_dim: the dimension of the input node features
        # output_dim: the dimension of the output node features
        # num_heads: the number of attention heads
        # activation: the activation function, default is None
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_heads=num_heads
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        # initialize the weight matrix and the attention vector
        self.weight=i.initializer([input_dim,num_heads*output_dim],weight_initializer,dtype)
        self.attention=i.initializer([num_heads,2*output_dim],bias_initializer,dtype)
        if use_bias==True:
            self.weight_list=[self.weight,self.bias]
        else:
            self.weight_list=[self.weight]
    
    
    def output(self,graph,data):
        # graph: a dictionary that represents the input graph structure data, containing adjacency matrix, node features, etc.
        # data: a tensor that represents the input node features, shape is [N, input_dim], where N is the number of nodes
        # compute the linear transformation and add bias, and split into multiple heads
        linear_output=tf.matmul(data,self.weight)
        linear_output=tf.reshape(linear_output,[-1,self.num_heads,self.output_dim])
        # compute the attention score, and normalize with softmax
        attention_input=tf.concat([linear_output[:,None,:,:],linear_output[None,:,:,:]],axis=-1)
        attention_score=tf.reduce_sum(self.attention*attention_input,axis=-1)
        attention_score=tf.where(graph["adj_matrix"]>0,attention_score,-1e9*tf.ones_like(attention_score))
        attention_score=tf.nn.softmax(attention_score,axis=-1)
        # compute the weighted aggregation of neighbor information, and concatenate multiple heads
        neighbor_output=tf.matmul(attention_score,linear_output)
        neighbor_output=tf.reshape(neighbor_output,[-1,self.num_heads*self.output_dim])
        # apply activation function
        if self.activation is not None:
            neighbor_output=self.activation(neighbor_output)
        return neighbor_output
