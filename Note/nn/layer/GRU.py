import tensorflow as tf # import the TensorFlow library
import Note.nn.initializer as i # import the initializer module from Note.nn package


class GRU: # define a class for gated recurrent unit (GRU) layer
    def __init__(self,output_size,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',return_sequence=False,use_bias=True,activation1=tf.nn.sigmoid,activation2=tf.nn.tanh,dtype='float32'): # define the constructor method
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.output_list=[] # initialize an empty list for output sequence
        self.state=tf.zeros(shape=[1,output_size],dtype=dtype) # initialize a zero vector for initial state
        self.H=tf.zeros(shape=[1,output_size],dtype=dtype) # initialize a zero vector for initial hidden state
        self.return_sequence=return_sequence # set the return sequence flag
        self.use_bias=use_bias # set the use bias flag
        self.dtype=dtype
        self.activation1=activation1 # set the activation function for gates (usually sigmoid)
        self.activation2=activation2 # set the activation function for candidate hidden state (usually tanh)
        self.output_size=output_size
        if input_size!=None:
            self.weight_r1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for reset gate input
            self.weight_r2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for reset gate hidden state
            self.weight_z1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for update gate input
            self.weight_z2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for update gate hidden state
            self.weight_h1=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix for candidate hidden state input
            self.weight_h2=i.initializer([output_size,output_size],weight_initializer,dtype) # initialize the weight matrix for candidate hidden state hidden state
            if use_bias==True: # if use bias is True
                self.bias_r=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for reset gate
                self.bias_z=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for update gate
                self.bias_h=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector for candidate hidden state
            if use_bias==True: # if use bias is True
                self.param=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2,self.bias_r,self.bias_z,self.bias_h] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2] # store only the weight matrices in a list
    
    
    def build(self):
        self.weight_r1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for reset gate input
        self.weight_r2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for reset gate hidden state
        self.weight_z1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for update gate input
        self.weight_z2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for update gate hidden state
        self.weight_h1=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for candidate hidden state input
        self.weight_h2=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for candidate hidden state hidden state
        if self.use_bias==True: # if use bias is True
            self.bias_r=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for reset gate
            self.bias_z=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for update gate
            self.bias_h=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector for candidate hidden state
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2,self.bias_r,self.bias_z,self.bias_h] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2] # store only the weight matrices in a list
        return
    
    
    def output(self,data): # define the output method
        timestep=data.shape[1] # get the number of timesteps from the input data shape
        if self.use_bias==True: # if use bias is True
            for j in range(timestep): # loop over the timesteps
                R=self.activation1(tf.matmul(data[:][:,j],self.weight_r1)+tf.matmul(self.state,self.weight_r2)+self.bias_r) # calculate the reset gate value by applying activation function to the linear transformation of input and state, plus bias
                Z=self.activation1(tf.matmul(data[:][:,j],self.weight_z1)+tf.matmul(self.state,self.weight_z2)+self.bias_z) # calculate the update gate value by applying activation function to the linear transformation of input and state, plus bias
                H_=self.activation2(tf.matmul(data[:][:,j],self.weight_h1)+tf.matmul(R*self.H,self.weight_h2)+self.bias_h) # calculate the candidate hidden state value by applying activation function to the linear transformation of input and reset gate multiplied by hidden state, plus bias
                output=Z*self.H+(1-Z)*H_ # calculate the output value by combining the update gate, hidden state and candidate hidden state values
                self.output_list.append(output) # append the output value to the output list
                self.H=output # update the hidden state value with the output value
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
                R=self.activation1(tf.matmul(data[:][:,j],self.weight_r1)+tf.matmul(self.state,self.weight_r2)) # calculate the reset gate value by applying activation function to the linear transformation of input and state
                Z=self.activation1(tf.matmul(data[:][:,j],self.weight_z1)+tf.matmul(self.state,self.weight_z2)) # calculate the update gate value by applying activation function to the linear transformation of input and state
                H_=self.activation2(tf.matmul(data[:][:,j],self.weight_h1)+tf.matmul(R*self.H,self.weight_h2)) # calculate the candidate hidden state value by applying activation function to the linear transformation of input and reset gate multiplied by hidden state
                output=Z*self.H+(1-Z)*H_ # calculate the output value by combining the update gate, hidden state and candidate hidden state values
                self.output_list.append(output) # append the output value to the output list
                self.H=output # update the hidden state value with the output value
                self.state=output # update the state value with the output value
            if self.return_sequence==True: # if return sequence is True
                output=tf.stack(self.output_list,axis=1) # stack the output list along the second dimension to form a tensor of shape [batch_size, timestep, hidden_size]
                self.output_list=[] # clear the output list
                return output # return the output tensor
            else: # if return sequence is False
                self.output_list=[] # clear the output list
                return output # return the last output value
