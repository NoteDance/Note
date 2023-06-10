import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class RNNCell:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True):
        self.weight_i=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_s=i.initializer(weight_shape,weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([weight_shape[1]],bias_initializer,dtype)
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight_i,self.weight_s,self.bias]
        else:
            self.weight_list=[self.weight_i,self.weight_s]
    
    
    def output(self,data,state):
        output=tf.matmul(data,self.weight_i)+tf.matmul(state,self.weight_s)
        if self.use_bias==True:
            output=output+self.bias
        if self.activation is not None:
            output=self.activation(output)
        return output
