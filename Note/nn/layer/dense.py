import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package


class dense: # define a class for dense (fully connected) layer
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True): # define the constructor method
        self.weight=i.initializer(weight_shape,weight_initializer,dtype) # initialize the weight matrix
        if use_bias==True: # if use bias is True
            self.bias=i.initializer([weight_shape[1]],bias_initializer,dtype) # initialize the bias vector
        else: # if use bias is False
            self.bias=None # set the bias to None
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.output_size=weight_shape[-1]
        if use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
    
    
    def output(self,data): # define the output method
        return a.activation(data,self.weight,self.bias,self.activation,self.use_bias) # return the output of applying activation function to the linear transformation of data and weight, plus bias if use bias is True
