import tensorflow as tf
from Note import nn


class SwitchNorm1d(nn.Layer):
    def __init__(self, input_size=None, eps=1e-5, momentum=0.997, using_moving_average=True, dtype='float32'):
        super().__init__()
        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.dtype = dtype
        if input_size!=None:
            self.gamma = nn.Parameter(tf.ones((1, input_size), dtype))
            self.gamma.name_ = 'weight'
            self.beta = nn.Parameter(tf.zeros((1, input_size), dtype))
            self.beta.name_ = 'bias'
            self.mean_weight = nn.Parameter(tf.ones(2))
            self.var_weight = nn.Parameter(tf.ones(2))
            self.moving_mean = nn.initializer_([1, input_size], 'zeros', dtype, trainable=False)
            self.moving_variance = nn.initializer_([1, input_size], 'zeros', dtype, trainable=False)
            nn.Model.param.append(self.moving_mean)
            nn.Model.param.append(self.moving_variance)
        nn.Model.register(self)
    
    def build(self):
        self.moving_mean = nn.initializer_([1, self.input_size], 'zeros', self.dtype, trainable=False)
        self.moving_variance = nn.initializer_([1, self.input_size], 'zeros', self.dtype, trainable=False)
        nn.Model.param.append(self.moving_mean)
        nn.Model.param.append(self.moving_variance)
        self.gamma = nn.Parameter(tf.ones((1, self.input_size), self.dtype))
        self.gamma.name_ = 'weight'
        self.beta = nn.Parameter(tf.zeros((1, self.input_size), self.dtype))
        self.beta.name_ = 'bias'
        self.mean_weight = nn.Parameter(tf.ones(2))
        self.var_weight = nn.Parameter(tf.ones(2))

    def _check_input_dim(self, input):
        if len(input.shape) != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(len(input.shape)))

    def __call__(self, x):
        self._check_input_dim(x)
        if x.dtype != self.dtype:
            x = tf.cast(x,self.dtype)
        if self.input_size == None:
            self.input_size = x.shape[-1]
            self.build()
        mean_ln = tf.reduce_mean(x, axis=1, keepdims=True)
        var_ln = tf.math.reduce_variance(x, axis=1, keepdims=True)

        if self.training:
            mean_bn = tf.reduce_mean(x, axis=0, keepdims=True)
            var_bn = tf.math.reduce_variance(x, axis=0, keepdims=True)
            if self.using_moving_average:
                self.moving_mean.assign(self.moving_mean * self.momentum)
                self.moving_mean.assign_add((1 - self.momentum) * mean_bn)
                self.moving_variance.assign(self.moving_variance * self.momentum)
                self.moving_variance.assign_add((1 - self.momentum) * var_bn)
            else:
                self.moving_mean.assign_add(mean_bn)
                self.moving_variance.assign_add(mean_bn.data ** 2 + var_bn)
        else:
            mean_bn = self.moving_mean
            var_bn = self.moving_variance

        mean_weight = tf.nn.softmax(self.mean_weight, axis=0)
        var_weight = tf.nn.softmax(self.var_weight, axis=0)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / tf.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SwitchNorm2d(nn.Layer):
    def __init__(self, input_size=None, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False, dtype='float32'):
        super().__init__()
        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.dtype = dtype
        if input_size!=None:
            self.gamma = nn.Parameter(tf.ones((1, 1, 1, input_size), dtype))
            self.gamma.name_ = 'weight'
            self.beta = nn.Parameter(tf.zeros((1, 1, 1, input_size), dtype))
            self.beta.name_ = 'bias'
            if self.using_bn:
                self.mean_weight = nn.Parameter(tf.ones(3))
                self.var_weight = nn.Parameter(tf.ones(3))
            else:
                self.mean_weight = nn.Parameter(tf.ones(2))
                self.var_weight = nn.Parameter(tf.ones(2))
            if self.using_bn:
                self.moving_mean = nn.initializer_([1, 1, input_size], 'zeros', dtype, trainable=False)
                self.moving_variance = nn.initializer_([1, 1, input_size], 'zeros', dtype, trainable=False)
                nn.Model.param.append(self.moving_mean)
                nn.Model.param.append(self.moving_variance)
        nn.Model.register(self)
    
    def build(self):
        if self.using_bn:
            self.moving_mean = nn.initializer_([1, 1, self.input_size], 'zeros', self.dtype, trainable=False)
            self.moving_variance = nn.initializer_([1, 1, self.input_size], 'zeros', self.dtype, trainable=False)
            nn.Model.param.append(self.moving_mean)
            nn.Model.param.append(self.moving_variance)
        self.gamma = nn.Parameter(tf.ones((1, 1, 1, self.input_size), self.dtype))
        self.gamma.name_ = 'weight'
        self.beta = nn.Parameter(tf.zeros((1, 1, 1, self.input_size), self.dtype))
        self.beta.name_ = 'bias'
        if self.using_bn:
            self.mean_weight = nn.Parameter(tf.ones(3))
            self.var_weight = nn.Parameter(tf.ones(3))
        else:
            self.mean_weight = nn.Parameter(tf.ones(2))
            self.var_weight = nn.Parameter(tf.ones(2))

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(len(input.shape)))

    def __call__(self, x):
        self._check_input_dim(x)
        if x.dtype != self.dtype:
            x = tf.cast(x,self.dtype)
        if self.input_size == None:
            self.input_size = x.shape[-1]
            self.build()
        N, H, W, C = x.shape
        x = tf.reshape(x, (N, -1, C))
        mean_in = tf.reduce_mean(x, axis=1, keepdims=True)
        var_in = tf.math.reduce_variance(x, axis=1, keepdims=True)

        mean_ln = tf.reduce_mean(mean_in, axis=-1, keepdims=True)
        temp = var_in + mean_in ** 2
        var_ln = tf.reduce_mean(temp, axis=-1, keepdims=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = tf.reduce_mean(mean_in, axis=0, keepdims=True)
                var_bn = tf.reduce_mean(temp, axis=0, keepdims=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.moving_mean.assign(self.moving_mean * self.momentum)
                    self.moving_mean.assign_add((1 - self.momentum) * mean_bn)
                    self.moving_variance.assign(self.moving_variance * self.momentum)
                    self.moving_variance.assign_add((1 - self.momentum) * var_bn)
                else:
                    self.moving_mean.assign_add(mean_bn)
                    self.moving_variance.assign_add(mean_bn ** 2 + var_bn)
            else:
                mean_bn = self.moving_mean
                var_bn = self.moving_variance

        mean_weight = tf.nn.softmax(self.mean_weight, axis=0)
        var_weight = tf.nn.softmax(self.var_weight, axis=0)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / tf.sqrt(var+self.eps)
        x = tf.reshape(x, (N, H, W, C))
        return x * self.weight + self.bias


class SwitchNorm3d(nn.Layer):
    def __init__(self, input_size=None, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False, dtype='float32'):
        super().__init__()
        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.dtype = dtype
        if input_size!=None:
            self.gamma = nn.Parameter(tf.ones((1, 1, 1, 1, input_size), dtype))
            self.gamma.name_ = 'weight'
            self.beta = nn.Parameter(tf.zeros((1, 1, 1, 1, input_size), dtype))
            self.beta.name_ = 'bias'
            if self.using_bn:
                self.mean_weight = nn.Parameter(tf.ones(3))
                self.var_weight = nn.Parameter(tf.ones(3))
            else:
                self.mean_weight = nn.Parameter(tf.ones(2))
                self.var_weight = nn.Parameter(tf.ones(2))
            if self.using_bn:
                self.moving_mean = nn.initializer_([1, 1, input_size], 'zeros', dtype, trainable=False)
                self.moving_variance = nn.initializer_([1, 1, input_size], 'zeros', dtype, trainable=False)
                nn.Model.param.append(self.moving_mean)
                nn.Model.param.append(self.moving_variance)
        nn.Model.register(self)
    
    def build(self):
        if self.using_bn:
            self.moving_mean = nn.initializer_([1, 1, self.input_size], 'zeros', self.dtype, trainable=False)
            self.moving_variance = nn.initializer_([1, 1, self.input_size], 'zeros', self.dtype, trainable=False)
            nn.Model.param.append(self.moving_mean)
            nn.Model.param.append(self.moving_variance)
        self.gamma = nn.Parameter(tf.ones((1, 1, 1, 1, self.input_size), self.dtype))
        self.gamma.name_ = 'weight'
        self.beta = nn.Parameter(tf.zeros((1, 1, 1, 1, self.input_size), self.dtype))
        self.beta.name_ = 'bias'
        if self.using_bn:
            self.mean_weight = nn.Parameter(tf.ones(3))
            self.var_weight = nn.Parameter(tf.ones(3))
        else:
            self.mean_weight = nn.Parameter(tf.ones(2))
            self.var_weight = nn.Parameter(tf.ones(2))

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(len(input.shape)))

    def __call__(self, x):
        self._check_input_dim(x)
        if x.dtype != self.dtype:
            x = tf.cast(x,self.dtype)
        if self.input_size == None:
            self.input_size = x.shape[-1]
            self.build()
        N, D, H, W, C = x.shape
        x = tf.reshape(x, (N, -1, C))
        mean_in = tf.reduce_mean(x, axis=1, keepdims=True)
        var_in = tf.math.reduce_variance(x, axis=1, keepdims=True)

        mean_ln = tf.reduce_mean(mean_in, axis=-1, keepdims=True)
        temp = var_in + mean_in ** 2
        var_ln = tf.reduce_mean(temp, axis=-1, keepdims=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = tf.reduce_mean(mean_in, axis=0, keepdims=True)
                var_bn = tf.reduce_mean(temp, axis=0, keepdims=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.moving_mean.assign(self.moving_mean * self.momentum)
                    self.moving_mean.assign_add((1 - self.momentum) * mean_bn)
                    self.moving_variance.assign(self.moving_variance * self.momentum)
                    self.moving_variance.assign_add((1 - self.momentum) * var_bn)
                else:
                    self.moving_mean.assign_add(mean_bn)
                    self.moving_variance.assign_add(mean_bn ** 2 + var_bn)
            else:
                mean_bn = self.moving_mean
                var_bn = self.moving_variance

        mean_weight = tf.nn.softmax(self.mean_weight, axis=0)
        var_weight = tf.nn.softmax(self.var_weight, axis=0)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, (N, D, H, W, C))
        return x * self.weight + self.bias