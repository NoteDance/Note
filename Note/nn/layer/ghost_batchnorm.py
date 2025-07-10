import tensorflow as tf
from Note import nn


class GhostBatchNorm(nn.Layer):
    def __init__(self, input_size, virtual_bs, momentum=0.9, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', dtype='float32'):
        super().__init__()
        self.input_size = input_size
        self.virtual_bs = virtual_bs
        self.momentum = momentum
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.dtype = dtype
        
        self.gamma = nn.Parameter(tf.ones(input_size, dtype=dtype))
        
        self.beta = nn.Parameter(tf.zeros(input_size, dtype=dtype))
        
        self.moving_mean = nn.initializer_([input_size], moving_mean_initializer, dtype, trainable=False)
        self.moving_variance = nn.initializer_([input_size], moving_variance_initializer, dtype, trainable=False)
        nn.Model.param.append(self.moving_mean)
        nn.Model.param.append(self.moving_variance)
        nn.Model.register(self)
    
    
    def build(self):
        self.moving_mean=nn.initializer_([self.input_size], self.moving_mean_initializer, self.dtype, trainable=False)
        self.moving_variance=nn.initializer_([self.input_size], self.moving_variance_initializer, self.dtype, trainable=False)
        nn.Model.param.append(self.moving_mean)
        nn.Model.param.append(self.moving_variance)
        self.gamma=nn.Parameter(tf.ones(self.input_size, dtype=self.dtype))
        self.beta=nn.Parameter(tf.zeros(self.input_size, dtype=self.dtype))
        return

    
    def __call__(self, X): 
        if X.dtype != self.dtype:
            X = tf.cast(X,self.dtype)
        if self.input_size == None:
            self.input_size = X.shape[-1]
            self.build()
        # on inference use running values
        if not self.training:
            return self.gamma * (X - self.running_mean)/self.running_std + self.beta
        
        # obtain ghost batches
        num_ghost_batches = tf.cast(tf.math.ceil(X.shape[0]/self.virtual_bs), tf.int32)
        ghost_batches = tf.reshape(X, (num_ghost_batches, self.virtual_bs, X.shape[-1]))
        
        # obtain metrics
        ghost_mean = tf.reduce_mean(ghost_batches, axis=1, keepdims=True)
        ghost_std = tf.math.reduce_std(ghost_batches, axis=1, keepdims=True)
        
        # normalize
        normalized_ghost_batches = (ghost_batches - ghost_mean) / ghost_std
        normalized_batch = tf.reshape(normalized_ghost_batches, X.shape)
        
        # update running metrics
        self.moving_mean.assign(self._calculate_running_metric(
            self.moving_mean, ghost_mean, num_ghost_batches))
        self.moving_variance.assign(self._calculate_running_metric(
            self.moving_variance, ghost_std, num_ghost_batches))
        
        return self.gamma * normalized_batch + self.beta
    
    def _calculate_running_metric(self, running_metric, ghost_metric, num_ghost_batches):
        weighted_prev = ((1-self.momentum)**num_ghost_batches) * running_metric
        
        exp_idxs = tf.range(num_ghost_batches)
        exp_idxs = tf.reverse(exp_idxs, axis=[0])
        weighted_new = tf.reduce_sum((
            tf.expand_dims((self.momentum * (1-self.momentum)**exp_idxs), axis=-1) * tf.squeeze(ghost_metric, axis=1)
        ), axis=0)
        
        return weighted_prev + weighted_new