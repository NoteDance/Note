import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class GAT:
    def __init__(self,input_dim,output_dim,num_heads,weight_initializer='Xavier',attention_initializer='zeros',activation=None,dtype='float32',attn_drop=0.0,ffd_drop=0.0):
        # input_dim: the dimension of the input node features
        # output_dim: the dimension of the output node features
        # num_heads: the number of attention heads
        # activation: the activation function, default is None
        # attn_drop: the dropout rate for attention scores, default is 0.0
        # ffd_drop: the dropout rate for features, default is 0.0
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_heads=num_heads
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        self.attn_drop=attn_drop
        self.ffd_drop=ffd_drop
        # initialize the weight matrix and the attention vector
        self.weight=i.initializer([input_dim,num_heads*output_dim],weight_initializer,dtype)
        self.attention=i.initializer([num_heads,2*output_dim],attention_initializer,dtype)
        self.param=[self.weight,self.attention]
    
    
    def output(self,graph,data):
        # graph: a dictionary that represents the input graph structure data, containing adjacency matrix, node features, etc.
        # data: a tensor that represents the input node features, shape is [N, input_dim], where N is the number of nodes
        # compute the linear transformation and add bias, and split into multiple heads
        linear_output=tf.matmul(data,self.weight)
        linear_output=tf.reshape(linear_output,[-1,self.num_heads,self.output_dim])
        # apply dropout to features
        linear_output=tf.nn.dropout(linear_output,self.ffd_drop)
        N=graph['adj_matrix'].shape[0]
        linear_output_1=tf.broadcast_to(linear_output[:,None,:,:],[N,N,self.num_heads,self.output_dim])
        linear_output_2=tf.broadcast_to(linear_output[None,:,:,:],[N,N,self.num_heads,self.output_dim])
        # compute the attention score, and normalize with softmax
        attention_input=tf.concat([linear_output_1,linear_output_2],axis=-1)
        attention_score=tf.reduce_sum(self.attention*attention_input,axis=-1)
        adj=tf.expand_dims(graph["adj_matrix"],axis=-1) # [N, N, 1]
        adj=tf.tile(adj,[1,1,self.num_heads]) # [N, N, num_heads]
        attention_score=tf.where(adj>0,attention_score,-1e9*tf.ones_like(attention_score))
        # apply LeakyReLU activation to attention score
        attention_score=tf.nn.leaky_relu(attention_score,alpha=0.2)
        attention_score=tf.nn.softmax(attention_score,axis=-1)
        # apply dropout to attention score
        attention_score=tf.nn.dropout(attention_score,self.attn_drop)
        # compute the weighted aggregation of neighbor information, and concatenate multiple heads
        neighbor_output=tf.matmul(attention_score,linear_output)
        neighbor_output=tf.reshape(neighbor_output,[-1,self.num_heads*self.output_dim])
        # apply activation function
        if self.activation is not None:
            neighbor_output=self.activation(neighbor_output)
        return neighbor_output