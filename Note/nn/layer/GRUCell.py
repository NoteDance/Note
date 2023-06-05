import tensorflow as tf
import Note.nn.initializer as i


class GRUCell:
    def __init__(self,weight_shape,weight_initializer='uniform',bias_initializer='zero',dtype='float32',use_bias=True,activation1=tf.nn.sigmoid,activation2=tf.nn.tanh):
        self.weight=i.initializer([weight_shape[0]+weight_shape[1],3*weight_shape[1]],weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([3*weight_shape[1]],bias_initializer,dtype)
        self.use_bias=use_bias
        self.activation1=activation1
        self.activation2=activation2
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
        r,z,h=tf.split(z,3,axis=-1)
        r=self.activation1(r)
        z=self.activation1(z)
        h=self.activation2(h)
        h_new=z*state+(1-z)*h
        output=h_new
        return output,h_new
