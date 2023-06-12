import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.LSTMCell import LSTMCell
from Note.nn.activation import activation_dict


'''
AutoLSTM is a neural network layer that can automatically adjust the hidden state dimension of 
a LSTM based on the input sequence length. It applies a length function to the input sequence to compute 
its effective length, and then applies a dimension function to map the length to the hidden state dimension. 
It also applies an init function to initialize the hidden state and cell state, and an lstm function to 
implement the lstm recurrence. The layer can also apply 
an activation function to the output features if specified.
'''
class AutoLSTM:
    def __init__(self,in_features,out_features,length_function=None,dimension_function=None,init_function=None,lstm_function=None,activation=None,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True):
        # initialize the auto lstm layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # length_function: the function to compute the effective length of the input sequence
        # dimension_function: the function to map the input sequence length to the hidden state dimension
        # init_function: the function to initialize the hidden state and cell state
        # lstm_function: the function to implement the lstm recurrence
        # activation: the activation function to apply to the output features
        # weight_initializer: the method to initialize the weight matrix for the lstm function
        # bias_initializer: the method to initialize the bias vector for the lstm function
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        self.param_list=[]
        if length_function is None:
            # use a default length function that counts non-zero elements along a given axis
            self.length_function=lambda x,axis=1:tf.math.count_nonzero(x,axis=axis)
        else:
            # use a user-defined length function
            self.length_function=length_function
        if dimension_function is None:
            # use a default dimension function that applies a linear transformation and a ReLU activation
            self.dimension_function=dense([1,1],weight_initializer=weight_initializer,bias_initializer=bias_initializer,activation='relu',dtype=dtype)
            self.weight_list.append(self.dimension_function.weight_list)
        else:
            # use a user-defined dimension function
            self.dimension_function=dimension_function
        self.param_list.append(self.dimension_function.weight_list)
        if init_function is None:
            # use a default init function that initializes hidden state and cell state randomly
            self.init_function=lambda x:(tf.random.normal([x.shape[0],x.shape[2]],dtype=x.dtype),tf.random.normal([x.shape[0],x.shape[2]],dtype=x.dtype))
        else:
            # use a user-defined init function
            self.init_function=init_function
        if lstm_function is None:
            # use a default lstm function that implements standard lstm formula with optional bias and activation
            self.lstm_function=LSTMCell([in_features,out_features],weight_initializer=weight_initializer,bias_initializer=bias_initializer,dtype=dtype,use_bias=use_bias)
        else:
            # use a user-defined lstm function
            self.lstm_function=lstm_function
        self.param_list.append(self.lstm_function.weight_list)
    
    
    def output(self,data):
        # define the output function to compute the output features from the input features
        # data: a tensor of shape [batch_size, seq_len, in_features]
        # return: a tensor of shape [batch_size, seq_len, out_features]
        output=[] # create an empty list to store output features
        length=self.length_function(data) # compute the effective length of input sequence
        dimension=self.dimension_function.output(tf.expand_dims(length,-1)) # compute the hidden state dimension by applying dimension function to length vector 
        data=tf.reshape(data,[data.shape[0],data.shape[1],dimension]) # reshape input features according to hidden state dimension
        hidden,cell=self.init_function(data) # initialize hidden state and cell state by applying init function to input features 
        for i in range(data.shape[1]): # for each time step 
            input=data[:,i,:] # get the input features at current time step 
            hidden,cell=self.lstm_function.output(input,(hidden,cell)) # update hidden state and cell state by applying lstm function to input features and previous states 
            output.append(hidden) # append hidden state to output list 
        output=tf.stack(output,axis=1) # convert output list to tensor 
        if self.activation is not None:
            output=self.activation(output) # apply activation function to output features 
        return output