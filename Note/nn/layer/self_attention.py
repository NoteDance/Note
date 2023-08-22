import tensorflow as tf # import the TensorFlow library
import Note.nn.initializer as i # import the initializer module from Note.nn package


class self_attention: # define a class for self-attention layer
    def __init__(self,weight_shape,weight_initializer='Xavier',dtype='float32'): # define the constructor method
        self.qw=i.initializer(weight_shape,weight_initializer,dtype) # initialize the weight matrix for query projection
        self.kw=i.initializer(weight_shape,weight_initializer,dtype) # initialize the weight matrix for key projection
        self.vw=i.initializer(weight_shape,weight_initializer,dtype) # initialize the weight matrix for value projection
        self.output_size=weight_shape[-1]
        self.param=[self.qw,self.kw,self.vw] # store the parameters in a list


    def output(self,data,a,mask=None): # define the output method
        query=tf.matmul(data,self.qw) # calculate the query vector by multiplying input data and query weight matrix
        key=tf.matmul(data,self.kw) # calculate the key vector by multiplying input data and key weight matrix
        value=tf.matmul(data,self.vw) # calculate the value vector by multiplying input data and value weight matrix
        query=tf.reshape(query,shape=[query.shape[0],query.shape[1],a,data.shape[2]//a]) # reshape the query vector into a tensor of shape [batch_size, timestep, number of heads, head size]
        key=tf.reshape(key,shape=[key.shape[0],key.shape[1],a,data.shape[2]//a]) # reshape the key vector into a tensor of shape [batch_size, timestep, number of heads, head size]
        value=tf.reshape(value,shape=[value.shape[0],value.shape[1],a,data.shape[2]//a]) # reshape the value vector into a tensor of shape [batch_size, timestep, number of heads, head size]
        query=tf.transpose(query,perm=[0,2,1,3]) # transpose the query tensor to swap the timestep and number of heads dimensions
        key=tf.transpose(key,perm=[0,2,1,3]) # transpose the key tensor to swap the timestep and number of heads dimensions
        value=tf.transpose(value,perm=[0,2,1,3]) # transpose the value tensor to swap the timestep and number of heads dimensions
        scores=tf.matmul(query,key,transpose_b=True)/tf.sqrt(data.shape[2]/a) # calculate the attention scores by multiplying query and key tensors (transposing key tensor along last two dimensions), and scaling by square root of head size
        if mask is not None: # if mask is not None
            scores+=mask*-1e9 # add a large negative value to the masked positions in the scores tensor
        attention_weights=tf.nn.softmax(scores,axis=-1) # apply softmax function to the scores tensor along the last dimension to get the attention weights
        output=tf.matmul(attention_weights,value) # calculate the output by multiplying attention weights and value tensors
        output=tf.transpose(output,perm=[0,2,1,3]) # transpose the output tensor to swap back the timestep and number of heads dimensions
        output=tf.reshape(output,shape=[output.shape[0],output.shape[1],-1]) # reshape the output tensor into a matrix of shape [batch_size, timestep, hidden_size]
        return output,attention_weights # return the output matrix and attention weights tensor
