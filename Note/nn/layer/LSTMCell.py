import tensorflow as tf
import Note.nn.initializer as i


class LSTMCell:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True,activation1=tf.nn.sigmoid,activation2=tf.nn.tanh):
        self.weight=i.initializer([weight_shape[0]+weight_shape[1],4*weight_shape[1]],weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([4*weight_shape[1]],bias_initializer,dtype)
        self.use_bias=use_bias
        self.activation1=activation1
        self.activation2=activation2
        if use_bias==True:
            self.param_list=[self.weight,self.bias]
        else:
            self.param_list=[self.weight]
    
    
    def output(self,data,state):
        x=tf.concat([data,state],axis=-1)
        if self.use_bias==True:
            z=tf.matmul(x,self.weight)+self.bias
        else:
            z=tf.matmul(x,self.weight)
        i,f,o,c=tf.split(z,4,axis=-1)
        i=self.activation1(i)
        f=self.activation1(f)
        o=self.activation1(o)
        c=self.activation2(c)
        c_new=i*c+f*state
        output=o*tf.nn.tanh(c_new)
        return output,c_new