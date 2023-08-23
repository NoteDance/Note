import tensorflow as tf
from Note.nn.initializer import initializer


class batch_normalization:
    def __init__(self, input_size, axes=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', dtype='float32'):
        self.axes=axes
        self.momentum=momentum
        self.epsilon=epsilon
        self.param=[]
        self.moving_mean=initializer([input_size], moving_mean_initializer, dtype)
        self.moving_var=initializer([input_size], moving_variance_initializer, dtype)
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
    
    
    def output(self, data, keepdims=False):
        mean, var = tf.nn.moments(data, self.axes, keepdims)
        self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1 - self.momentum)) 
        self.moving_var.assign(self.moving_var * self.momentum + var * (1 - self.momentum))
        output = tf.nn.batch_normalization(data,
                                           mean=self.moving_mean,
                                           variance=self.moving_var,
                                           offset=self.beta,
                                           scale=self.gamma,
                                           variance_epsilon=self.epsilon)
        return output