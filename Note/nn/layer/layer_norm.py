import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Module import Module


class layer_norm:
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', dtype='float32'):
        self.input_size=input_size
        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.dtype=dtype
        if input_size!=None:
            self.output_size=input_size
            self.param=[]
            if center==True:
                self.beta=initializer([input_size], beta_initializer, dtype)
                self.param.append(self.beta)
            else:
                self.beta=None
            if scale==True:
                self.gamma=initializer([input_size], gamma_initializer, dtype)
                self.param.append(self.gamma)
            else:
                self.gamma=None
            Module.param.extend(self.param)
    
    
    def build(self):
        self.output_size=self.input_size
        self.param=[]
        if self.center==True:
            self.beta=initializer([self.input_size], self.beta_initializer, self.dtype)
            self.param.append(self.beta)
        else:
            self.beta=None
        if self.scale==True:
            self.gamma=initializer([self.input_size], self.gamma_initializer, self.dtype)
            self.param.append(self.gamma)
        else:
            self.gamma=None
        Module.param.extend(self.param)
        return
    
    
    def output(self, data):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        mean, variance = tf.nn.moments(data, self.axis, keepdims=True)
        output = tf.nn.batch_normalization(
            data,
            mean,
            variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=self.epsilon,
        )
        return output
