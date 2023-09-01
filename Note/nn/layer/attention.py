import tensorflow as tf # import the TensorFlow library
import Note.nn.initializer as i # import the initializer module from Note.nn package


class attention: # define a class for attention mechanism
    def __init__(self,weight_shape,weight_initializer='Xavier',trainable=True,dtype='float32'): # define the constructor method
        self.qw=i.initializer(weight_shape,weight_initializer,dtype) # initialize the query weight matrix
        self.kw=i.initializer(weight_shape,weight_initializer,dtype) # initialize the key weight matrix
        self.sw=i.initializer([weight_shape[1],1],weight_initializer,dtype) # initialize the score weight vector
        if trainable==True:
            self.param=[self.qw,self.kw,self.sw] # store the parameters in a list
        else:
            self.param=[]
    
    
    def output(self,en_h,de_h,score_en_h=None): # define the output method
        if score_en_h is None: # if no previous score is given
            score_en_h=tf.matmul(en_h,self.qw) # calculate the score by multiplying encoder hidden state and query weight matrix
        score=tf.matmul(tf.nn.tanh(score_en_h+tf.expand_dims(tf.matmul(de_h,self.kw),axis=1)),self.sw) # calculate the score by adding the previous score and the product of decoder hidden state and key weight matrix, then applying tanh activation and multiplying by score weight vector
        attention_weights=tf.nn.softmax(score,axis=1) # apply softmax function to get the attention weights
        if len(en_h.shape)==2: # if encoder hidden state is a 2D tensor
            context_vector=tf.squeeze(tf.matmul(tf.transpose(en_h,[1,0]),attention_weights),axis=-1) # calculate the context vector by multiplying the transpose of encoder hidden state and attention weights, then squeezing the last dimension
        else: # if encoder hidden state is a 3D tensor
            context_vector=tf.squeeze(tf.matmul(tf.transpose(en_h,[0,2,1]),attention_weights),axis=-1) # calculate the context vector by multiplying the transpose of encoder hidden state and attention weights along the second and third dimensions, then squeezing the last dimension
        attention_weights=tf.squeeze(attention_weights,axis=-1) # squeeze the last dimension of attention weights
        return context_vector,score_en_h,attention_weights # return the context vector, score, and attention weights
