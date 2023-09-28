import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package
from Note.nn.Module import Module


class dense(Module): # define a class for dense (fully connected) layer
    def __init__(self,output_size,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=output_size
        if input_size!=None:
            self.weight=i.initializer([input_size,output_size],weight_initializer,dtype) # initialize the weight matrix
            if use_bias==True: # if use bias is True
                self.bias=i.initializer([output_size],bias_initializer,dtype) # initialize the bias vector
            else: # if use bias is False
                self.bias=None # set the bias to None
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            if trainable==False:
                self.param=[]
            Module.param.extend(self.param)
    
    
    def build(self):
        self.weight=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix
        if self.use_bias==True: # if use bias is True
            self.bias=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector
        else: # if use bias is False
            self.bias=None # set the bias to None
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
        if self.trainable==False:
            self.param=[]
        Module.param.extend(self.param)
        return
    
    
    def output(self,data): # define the output method
        return a.activation(data,self.weight,self.bias,self.activation,self.use_bias) # return the output of applying activation function to the linear transformation of data and weight, plus bias if use bias is True
