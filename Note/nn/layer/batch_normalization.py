import tensorflow as tf
from Note.nn.initializer import initializer
from multiprocessing import Manager
from Note.nn.Module import Module


class batch_normalization:
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', keepdims=True, trainable=True, parallel=True, dtype='float32'):
        self.input_size=input_size
        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.keepdims=keepdims
        self.trainable=trainable
        self.parallel=parallel
        self.dtype=dtype
        self.train_flag=True
        if input_size!=None:
            self.output_size=input_size
            self.moving_mean=tf.zeros([input_size],dtype)
            self.moving_var=tf.ones([input_size],dtype)
            if parallel:
                manager=Manager()
                self.moving_mean=manager.list([self.moving_mean])
                self.moving_var=manager.list([self.moving_var])
                Module.ctl_list.append(self.convert_to_list)
                Module.ctsl_list.append(self.convert_to_shared_list)
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
        if self.parallel:
            manager=Manager()
            self.moving_mean=manager.list([self.moving_mean])
            self.moving_var=manager.list([self.moving_var])
            Module.ctl_list.append(self.convert_to_list)
            Module.ctsl_list.append(self.convert_to_shared_list)
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
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        self.train_flag=train_flag
        if self.train_flag:
            mean, var = tf.nn.moments(data, self.axis, keepdims=self.keepdims)
            if self.parallel:
                self.moving_mean[0]=self.moving_mean[0] * self.momentum + mean * (1 - self.momentum)
                self.moving_var[0]=self.moving_var[0] * self.momentum + var * (1 - self.momentum)
            else:
                self.moving_mean=self.moving_mean * self.momentum + mean * (1 - self.momentum)
                self.moving_var=self.moving_var * self.momentum + var * (1 - self.momentum)
            output = tf.nn.batch_normalization(data,
                                               mean=mean,
                                               variance=var,
                                               offset=self.beta,
                                               scale=self.gamma,
                                               variance_epsilon=self.epsilon)
        else:
            if self.parallel:
                output = tf.nn.batch_normalization(data,
                                   mean=self.moving_mean[0],
                                   variance=self.moving_var[0],
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
    
    
    def convert_to_list(self):
        self.moving_mean=list(self.moving_mean)
        self.moving_var=list(self.moving_var)
        return
    
    
    def convert_to_shared_list(self,manager):
        self.moving_mean=manager.list(self.moving_mean)
        self.moving_var=manager.list(self.moving_var)
        return
