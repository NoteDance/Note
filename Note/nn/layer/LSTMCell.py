import tensorflow as tf # import the TensorFlow library
import Note.nn.initializer as i # import the initializer module from Note.nn package


class LSTMCell: # define a class for long short-term memory (LSTM) cell
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True,activation1=tf.nn.sigmoid,activation2=tf.nn.tanh): # define the constructor method
        self.weight=i.initializer([weight_shape[0]+weight_shape[1],4*weight_shape[1]],weight_initializer,dtype) # initialize the weight matrix for all gates and candidate cell state
        if use_bias==True: # if use bias is True
            self.bias=i.initializer([4*weight_shape[1]],bias_initializer,dtype) # initialize the bias vector for all gates and candidate cell state
        self.use_bias=use_bias # set the use bias flag
        self.activation1=activation1 # set the activation function for gates (usually sigmoid)
        self.activation2=activation2 # set the activation function for candidate cell state (usually tanh)
        self.output_size=weight_shape[-1]
        if use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight matrix in a list
    
    
    def output(self,data,state): # define the output method
        x=tf.concat([data,state],axis=-1) # concatenate the input data and state along the last dimension
        if self.use_bias==True: # if use bias is True
            z=tf.matmul(x,self.weight)+self.bias # calculate the linear transformation of concatenated data and weight matrix, plus bias vector
        else: # if use bias is False
            z=tf.matmul(x,self.weight) # calculate the linear transformation of concatenated data and weight matrix
        i,f,o,c=tf.split(z,4,axis=-1) # split the linear transformation into four parts: input gate, forget gate, output gate and candidate cell state
        i=self.activation1(i) # apply activation function to the input gate
        f=self.activation1(f) # apply activation function to the forget gate
        o=self.activation1(o) # apply activation function to the output gate
        c=self.activation2(c) # apply activation function to the candidate cell state
        c_new=i*c+f*state # calculate the new cell state value by combining the input gate, candidate cell state and forget gate multiplied by previous state values
        output=o*tf.nn.tanh(c_new) # calculate the output value by multiplying the output gate and the tanh activation of the new cell state value
        return output,c_new # return the output value and the new cell state value
