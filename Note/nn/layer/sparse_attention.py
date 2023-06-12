import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict


class sparse_attention:
    def __init__(self,input_dim,output_dim,num_heads,weight_initializer='Xavier',bias_initializer='zeros',activation='relu',dtype='float32',use_bias=True):
        # input_dim: the dimension of the input features
        # output_dim: the dimension of the output features
        # num_heads: the number of attention heads to use
        # weight_initializer: the initializer for the weight matrices
        # bias_initializer: the initializer for the bias vectors
        # activation: the activation function for the attention scores, default is None
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_heads=num_heads
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
         # initialize the weight matrices and the bias vectors for the query, key and value projections for each head
        self.weight_Q=[]
        self.bias_Q=[]
        self.weight_K=[]
        self.bias_K=[]
        self.weight_V=[]
        self.bias_V=[]
        for i in range(num_heads):
            self.weight_Q.append(initializer([input_dim,output_dim//num_heads],weight_initializer,dtype))
            self.bias_Q.append(initializer([output_dim//num_heads],bias_initializer,dtype))
            self.weight_K.append(initializer([input_dim,output_dim//num_heads],weight_initializer,dtype))
            self.bias_K.append(initializer([output_dim//num_heads],bias_initializer,dtype))
            self.weight_V.append(initializer([input_dim,output_dim//num_heads],weight_initializer,dtype))
            self.bias_V.append(initializer([output_dim//num_heads],bias_initializer,dtype))
        if use_bias==True:
            self.param_list=self.weight_Q+self.bias_Q+self.weight_K+self.bias_K+self.weight_V+self.bias_V
        else:
            self.param_list=self.weight_Q+self.weight_K+self.weight_V
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, input_dim], where N is the number of samples
        # create a list to store the outputs of each head
        outputs=[]
        for i in range(self.num_heads):
            # compute the query, key and value projections for the i-th head
            query=tf.matmul(data,self.weight_Q[i])
            query=tf.nn.bias_add(query,self.bias_Q[i])
            key=tf.matmul(data,self.weight_K[i])
            key=tf.nn.bias_add(key,self.bias_K[i])
            value=tf.matmul(data,self.weight_V[i])
            value=tf.nn.bias_add(value,self.bias_V[i])
            # compute the attention scores by dot product of query and key, and apply activation function if any
            scores=tf.matmul(query,key,transpose_b=True)
            if self.activation is not None:
                scores=self.activation(scores)
            # normalize the scores by softmax function
            scores=tf.nn.softmax(scores,axis=-1)
            # compute the output by weighted sum of value and scores
            output=tf.matmul(scores,value)
            # append the output to the list
            outputs.append(output)
        # concatenate the outputs of each head along the last dimension
        output=tf.concat(outputs,axis=-1)
        return output