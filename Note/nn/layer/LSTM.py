import tensorflow as tf # import the TensorFlow library
import Note.nn.initializer as i # import the initializer module from Note.nn package
from Note.nn.Model import Model


class LSTM: # define a class for long short-term memory (LSTM) layer
    def __init__(self,output_size,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',return_sequence=False,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.output_list=[] # initialize an empty list for output sequence
        self.state=tf.zeros(shape=[1,output_size],dtype=dtype) # initialize a zero vector for initial state
        self.C=tf.zeros(shape=[1,output_size],dtype=dtype) # initialize a zero vector for initial cell state
        self.return_sequence=return_sequence # set the return sequence flag
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=output_size
        if input_size!=None:
            self.weight_i1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for input gate input
            self.weight_i2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for input gate hidden state
            self.weight_f1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for forget gate input
            self.weight_f2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for forget gate hidden state
            self.weight_o1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for output gate input
            self.weight_o2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for output gate hidden state
            self.weight_c1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for candidate cell state input
            self.weight_c2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for candidate cell state hidden state
            if use_bias==True: # if use bias is True
                self.bias_i=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for input gate
                self.bias_f=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for forget gate
                self.bias_o=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for output gate
                self.bias_c=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for candidate cell state
            if trainable==True:
                if use_bias==True: # if use bias is True
                    self.param=[self.weight_i1,self.weight_f1,self.weight_o1,self.weight_c1,self.weight_i2,self.weight_f2,self.weight_o2,self.weight_c2,self.bias_i,self.bias_f,self.bias_o,self.bias_c] # store the parameters in a list
                else: # if use bias is False
                    self.param=[self.weight_i1,self.weight_f1,self.weight_o1,self.weight_c1,self.weight_i2,self.weight_f2,self.weight_o2,self.weight_c2] # store only the weight matrices in a list
            else:
                self.param=[]
            Model.param.extend(self.param)
    
    
    def build(self):
        self.weight_i1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for input gate input
        self.weight_i2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for input gate hidden state
        self.weight_f1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for forget gate input
        self.weight_f2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for forget gate hidden state
        self.weight_o1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for output gate input
        self.weight_o2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for output gate hidden state
        self.weight_c1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for candidate cell state input
        self.weight_c2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for candidate cell state hidden state
        if self.use_bias==True: # if use bias is True
            self.bias_i=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for input gate
            self.bias_f=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for forget gate
            self.bias_o=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for output gate
            self.bias_c=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for candidate cell state
        if self.trainable==True:
            if self.use_bias==True: # if use bias is True
                self.param=[self.weight_i1,self.weight_f1,self.weight_o1,self.weight_c1,self.weight_i2,self.weight_f2,self.weight_o2,self.weight_c2,self.bias_i,self.bias_f,self.bias_o,self.bias_c] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight_i1,self.weight_f1,self.weight_o1,self.weight_c1,self.weight_i2,self.weight_f2,self.weight_o2,self.weight_c2] # store only the weight matrices in a list
        else:
            self.param=[]
        Model.param.extend(self.param)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        timestep=data.shape[1] # get the number of timesteps from the input data shape
        if self.use_bias==True: # if use bias is True
            for j in range(timestep): # loop over the timesteps
                I=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_i1)+tf.matmul(self.state,self.weight_i2)+self.bias_i) # calculate the input gate value by applying activation function to the linear transformation of input and state, plus bias
                F=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_f1)+tf.matmul(self.state,self.weight_f2)+self.bias_f) # calculate the forget gate value by applying activation function to the linear transformation of input and state, plus bias
                O=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_o1)+tf.matmul(self.state,self.weight_o2)+self.bias_o) # calculate the output gate value by applying activation function to the linear transformation of input and state, plus bias
                C_=tf.nn.tanh(tf.matmul(data[:][:,j],self.weight_c1)+tf.matmul(self.state,self.weight_c2)+self.bias_c) # calculate the candidate cell state value by applying activation function to the linear transformation of input and state, plus bias
                C=I*C_+self.C*F # calculate the new cell state value by combining the input gate, candidate cell state and forget gate multiplied by previous cell state values
                output=O*tf.nn.tanh(C) # calculate the output value by multiplying the output gate and the tanh activation of the new cell state value
                self.output_list.append(output) # append the output value to the output list
                self.C=C # update the cell state value with the new cell state value
                self.state=output # update the state value with the output value
            if self.return_sequence==True: # if return sequence is True
                output=tf.stack(self.output_list,axis=1) # stack the output list along the second dimension to form a tensor of shape [batch_size, timestep, hidden_size]
                self.output_list=[] # clear the output list
                return output # return the output tensor
            else: # if return sequence is False
                self.output_list=[] # clear the output list
                return output # return the last output value
        else: # if use bias is False
            for j in range(timestep): # loop over the timesteps
                I=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_i1)+tf.matmul(self.state,self.weight_i2)) # calculate the input gate value by applying activation function to the linear transformation of input and state
                F=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_f1)+tf.matmul(self.state,self.weight_f2)) # calculate the forget gate value by applying activation function to the linear transformation of input and state
                O=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_o1)+tf.matmul(self.state,self.weight_o2)) # calculate the output gate value by applying activation function to the linear transformation of input and state
                C_=tf.nn.tanh(tf.matmul(data[:][:,j],self.weight_c1)+tf.matmul(self.state,self.weight_c2)) # calculate the candidate cell state value by applying activation function to the linear transformation of input and state
                C=I*C_+self.C*F # calculate the new cell state value by combining the input gate, candidate cell state and forget gate multiplied by previous cell state values
                output=O*tf.nn.tanh(C) # calculate the output value by multiplying the output gate and the tanh activation of the new cell state value
                self.output_list.append(output) # append the output value to the output list
                self.C=C # update the cell state value with the new cell state value
                self.state=output # update the state value with the output value
            if self.return_sequence==True: # if return sequence is True
                output=tf.stack(self.output_list,axis=1) # stack the output list along the second dimension to form a tensor of shape [batch_size, timestep, hidden_size]
                self.output_list=[] # clear the output list
                return output # return the output tensor
            else: # if return sequence is False
                self.output_list=[] # clear the output list
                return output # return the last output value