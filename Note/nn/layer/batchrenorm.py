import tensorflow as tf
from Note import nn


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(nn.Layer):
    def __init__(self, input_size=None, epsilon=1e-3, momentum=0.99, affine=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', dtype='float32'):
        super().__init__()
        self.input_size = input_size
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.dtype = dtype
        
        if input_size!=None:
            self.gamma = nn.Parameter(tf.ones(input_size, dtype=dtype))
            self.gamma.name_ = 'weight'
        
            self.beta = nn.Parameter(tf.zeros(input_size, dtype=dtype))
            self.beta.name_ = 'bias'
        
            self.moving_mean = nn.initializer_([input_size], moving_mean_initializer, dtype, trainable=False)
            self.moving_variance = nn.initializer_([input_size], moving_variance_initializer, dtype, trainable=False)
            nn.Model.param.append(self.moving_mean)
            nn.Model.param.append(self.moving_variance)
            
        nn.Model.register(self)
        
        self.num_batches_tracked = tf.Variable(tf.zeros((), dtype=tf.int64))
    
    def build(self):
        self.moving_mean = nn.initializer_([self.input_size], self.moving_mean_initializer, self.dtype, trainable=False)
        self.moving_variance = nn.initializer_([self.input_size], self.moving_variance_initializer, self.dtype, trainable=False)
        nn.Model.param.append(self.moving_mean)
        nn.Model.param.append(self.moving_variance)
        self.gamma = nn.Parameter(tf.ones(self.input_size, dtype=self.dtype))
        self.gamma.name_ = 'weight'
        self.beta = nn.Parameter(tf.zeros(self.input_size, dtype=self.dtype))
        self.beta.name_ = 'bias'
        self.num_batches_tracked = tf.Variable(tf.zeros((), dtype=tf.int64))
        return
    
    def _check_input_dim(self, x: tf.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> tf.Tensor:
        return tf.clip_by_value(2.0 / 35000.0 * self.num_batches_tracked + 25.0 / 35.0, clip_value_min=1.0, clip_value_max=3.0)

    @property
    def dmax(self) -> tf.Tensor:
        return tf.clip_by_value(5.0 / 20000.0 * self.num_batches_tracked - 25.0 / 20.0, clip_value_min=0.0, clip_value_max=5.0)
    
    def __call__(self, x: tf.Tensor, mask = None) -> tf.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        if x.dtype != self.dtype:
            x = tf.cast(x,self.dtype)
        if self.input_size == None:
            self.input_size = x.shape[-1]
            self.build()
        if self.training:
            dims = [i for i in range(len(x.shape) - 1)]
            if mask is not None:
                z = tf.boolean_mask(x, ~mask)
                batch_mean = tf.reduce_mean(z, axis=0)
                batch_std = tf.math.reduce_std(z, axis=0, keepdims=False) + self.epsilon
            else:
                batch_mean = tf.reduce_mean(x, axis=dims)
                batch_std = tf.math.reduce_std(x, axis=dims, keepdims=False) + self.epsilon
            
            r = tf.clip_by_value(
                    batch_std / tf.reshape(self.moving_variance, batch_std.shape),
                    clip_value_min=1 / self.rmax,
                    clip_value_max=self.rmax)
            d = tf.clip_by_value(
                (batch_mean - tf.reshape(self.moving_mean, batch_std.shape)) / 
                tf.reshape(self.moving_variance, batch_std.shape),
                clip_value_min=-self.dmax,
                clip_value_max=self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.moving_mean.assign_add(self.momentum * (
                batch_mean - self.moving_mean
            ))
            self.moving_variance.assign_add(self.momentum * (
                batch_std - self.moving_variance
            ))
            self.num_batches_tracked.assign_add(1)
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.gamma * x + self.beta
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: tf.Tensor) -> None:
        if len(x.shape) not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: tf.Tensor) -> None:
        if len(x.shape) != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: tf.Tensor) -> None:
        if len(x.shape) != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")