import tensorflow as tf
import Note.nn.initializer as i


class TransformerLayer:
    def __init__(self,weight_shape,weight_initializer='uniform',bias_initializer='zero',dtype='float64',use_bias=True):
        self.weight_q=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_k=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_v=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_o=i.initializer(weight_shape,weight_initializer,dtype)
        if use_bias==True:
            self.bias_q=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_k=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_v=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_o=i.initializer([weight_shape[1]],bias_initializer,dtype)
        self.dtype=dtype
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight_q,self.weight_k,self.weight_v,self.weight_o,self.bias_q,self.bias_k,self.bias_v,self.bias_o]
        else:
            self.weight_list=[self.weight_q,self.weight_k,self.weight_v,self.weight_o]
    
    
    def output(self,data):
        if self.use_bias==True:
            q=tf.matmul(data,self.weight_q)+self.bias_q
            k=tf.matmul(data,self.weight_k)+self.bias_k
            v=tf.matmul(data,self.weight_v)+self.bias_v
        else:
            q=tf.matmul(data,self.weight_q)
            k=tf.matmul(data,self.weight_k)
            v=tf.matmul(data,self.weight_v)
        w=tf.nn.softmax(tf.matmul(q,tf.transpose(k))/tf.math.sqrt(tf.cast(k.shape[-1],self.dtype)))
        o=tf.matmul(w,v)
        if self.use_bias==True:
            output=tf.matmul(o,self.weight_o)+self.bias_o
        else:
            output=tf.matmul(o,self.weight_o)
        return output