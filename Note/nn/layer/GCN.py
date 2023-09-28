import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict
from Note.nn.Module import Module


class GCN(Module):
    def __init__(self,input_dim,output_dim,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True):
        # input_dim: the dimension of the input node features
        # output_dim: the dimension of the output node features
        # activation: the activation function, default is None
        self.input_dim=input_dim
        self.output_dim=output_dim
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        # initialize the weight matrix and the bias vector
        self.weight=i.initializer([input_dim, output_dim],weight_initializer,dtype)
        self.bias=i.initializer([output_dim],bias_initializer,dtype)
        self.dtype=dtype
        self.output_size=output_dim
        if use_bias==True:
            self.param=[self.weight,self.bias]
        else:
            self.param=[self.weight]
        Module.param.extend(self.param)
    
    
    def output(self,graph,data):
        # graph: a dictionary that represents the input graph structure data, containing adjacency matrix, node features, etc.
        # data: a tensor that represents the input node features, shape is [N, input_dim], where N is the number of nodes
        # get the adjacency matrix, and normalize it
        adj_matrix=graph["adj_matrix"]
        degree_matrix=tf.reduce_sum(adj_matrix,axis=1)
        degree_matrix_inv_sqrt=tf.linalg.diag(tf.pow(tf.cast(degree_matrix,dtype=self.dtype)+1e-10,-0.5))
        norm_adj_matrix=tf.matmul(degree_matrix_inv_sqrt,
                                    tf.matmul(tf.cast(adj_matrix,dtype=self.dtype),degree_matrix_inv_sqrt))
        # compute the linear transformation and add bias
        linear_output=tf.matmul(data,self.weight)
        linear_output=tf.nn.bias_add(linear_output,self.bias)
        # compute the aggregation of neighbor information
        neighbor_output=tf.matmul(norm_adj_matrix,linear_output)
        # apply activation function
        if self.activation is not None:
            neighbor_output=self.activation(neighbor_output)
        return neighbor_output