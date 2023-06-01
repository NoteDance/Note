import tensorflow as tf
import Note.nn.initializer as i


class LSTMCell:
    def __init__(self,weight_shape,weight_initializer='uniform',bias_initializer='zero',dtype='float64',use_bias=True):
        self.weight=i.initializer([weight_shape[0]+weight_shape[1],4*weight_shape[1]],weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([4*weight_shape[1]],bias_initializer,dtype)
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight,self.bias]
        else:
            self.weight_list=[self.weight]
    
    
    def output(self,data,state):
        x=tf.concat([data,state],axis=-1)
        if self.use_bias==True:
            z=tf.matmul(x,self.weight)+self.bias
        else:
            z=tf.matmul(x,self.weight)
        i,f,o,c=tf.split(z,4,axis=-1)
        i=tf.nn.sigmoid(i)
        f=tf.nn.sigmoid(f)
        o=tf.nn.sigmoid(o)
        c=tf.nn.tanh(c)
        c_new=i*c+f*state
        output=o*tf.nn.tanh(c_new)
        return output,c_new