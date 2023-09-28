import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Module import Module


class batch_normalization(Module):
    def __init__(self, input_size=None, axes=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', keepdims=False, trainable=True, dtype='float32'):
        self.input_size=input_size
        self.axes=axes
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.keepdims=keepdims
        self.trainable=trainable
        self.dtype=dtype
        self.train_flag=True
        if input_size!=None:
            self.output_size=input_size
            self.moving_mean=tf.zeros([input_size],dtype)
            self.moving_var=tf.ones([input_size],dtype)
            self.param=[]
            if center==True:
                self.beta=initializer([input_size], beta_initializer, dtype)
                if trainable==True:
                    self.param.append(self.beta)
            else:
                self.beta=None
            if scale==True:
                self.gamma=initializer([input_size], gamma_initializer, dtype)
                if trainable==True:
                    self.param.append(self.gamma)
            else:
                self.gamma=None
            Module.param.extend(self.param)
    
    
    def build(self):
        self.output_size=self.input_size
        self.moving_mean=tf.zeros([self.input_size],self.dtype)
        self.moving_var=tf.ones([self.input_size],self.dtype)
        self.param=[]
        if self.center==True:
            self.beta=initializer([self.input_size], self.beta_initializer, self.dtype)
            if self.trainable==True:
                self.param.append(self.beta)
        else:
            self.beta=None
        if self.scale==True:
            self.gamma=initializer([self.input_size], self.gamma_initializer, self.dtype)
            if self.trainable==True:
                self.param.append(self.gamma)
        else:
            self.gamma=None
        Module.param.extend(self.param)
        return
    
    
    def output(self, data, train_flag=True):
        self.train_flag=train_flag
        if self.train_flag:
            mean, var = tf.nn.moments(data, self.axes, keepdims=self.keepdims)
            self.moving_mean=self.moving_mean * self.momentum + mean * (1 - self.momentum)
            self.moving_var=self.moving_var * self.momentum + var * (1 - self.momentum)
            output = tf.nn.batch_normalization(data,
                                               mean=mean,
                                               variance=var,
                                               offset=self.beta,
                                               scale=self.gamma,
                                               variance_epsilon=self.epsilon)
        else:
            output = tf.nn.batch_normalization(data,
                                               mean=self.moving_mean,
                                               variance=self.moving_var,
                                               offset=self.beta,
                                               scale=self.gamma,
                                               variance_epsilon=self.epsilon)
        return output
