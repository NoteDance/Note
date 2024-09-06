import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Model import Model


class layer_norm:
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, rms_scaling=False, beta_initializer='zeros', gamma_initializer='ones', dtype='float32'):
        self.input_size=input_size
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
            self.axis = [self.axis]
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.rms_scaling=rms_scaling
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.dtype=dtype
        self.input_shape=None
        if input_size!=None:
            self.output_size=input_size
            self.param=[]
            if center==True:
                self.beta=initializer([input_size], beta_initializer, dtype)
                self.param.append(self.beta)
                if Model.name!=None and Model.name not in Model.layer_param:
                    Model.layer_param[Model.name]=[]
                    Model.layer_param[Model.name].append(self.beta)
                elif Model.name!=None:
                    Model.layer_param[Model.name].append(self.beta)
            else:
                self.beta=None
            if scale==True:
                self.gamma=initializer([input_size], gamma_initializer, dtype)
                self.param.append(self.gamma)
                if Model.name!=None and Model.name not in Model.layer_param:
                    Model.layer_param[Model.name]=[]
                    Model.layer_param[Model.name].append(self.gamma)
                elif Model.name!=None:
                    Model.layer_param[Model.name].append(self.gamma)
            else:
                self.gamma=None
    
    
    def build(self):
        self.output_size=self.input_size
        if isinstance(self.axis, list):
            shape = tuple([self.input_shape[dim] for dim in self.axis])
        else:
            shape = (self.input_shape[self.axis],)
            self.axis = [self.axis]
        self.param=[]
        if self.center==True:
            self.beta=initializer(shape, self.beta_initializer, self.dtype)
            self.param.append(self.beta)
            if Model.name!=None and Model.name not in Model.layer_param:
                Model.layer_param[Model.name]=[]
                Model.layer_param[Model.name].append(self.beta)
            elif Model.name!=None:
                Model.layer_param[Model.name].append(self.beta)
        else:
            self.beta=None
        if self.scale==True:
            self.gamma=initializer(shape, self.gamma_initializer, self.dtype)
            self.param.append(self.gamma)
            if Model.name!=None and Model.name not in Model.layer_param:
                Model.layer_param[Model.name]=[]
                Model.layer_param[Model.name].append(self.gamma)
            elif Model.name!=None:
                Model.layer_param[Model.name].append(self.gamma)
        else:
            self.gamma=None
        return
    
    
    def __call__(self, data):
        # Compute the axes along which to reduce the mean / variance
        input_shape = data.shape
        ndims = len(input_shape)

        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.input_shape=input_shape
            self.build()

        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape[dim]

        def _broadcast(v):
            if (
                v is not None
                and len(v.shape) != ndims
                and self.axis != [ndims - 1]
            ):
                return tf.reshape(v, broadcast_shape)
            return v

        if self.rms_scaling:
            # Calculate outputs with only variance and gamma if rms scaling
            # is enabled
            # Calculate the variance along self.axis (layer activations).
            variance = tf.math.reduce_variance(data, axis=self.axis, keepdims=True)
            inv = tf.math.rsqrt(variance + self.epsilon)

            outputs = data * inv * tf.cast(self.gamma, data.dtype)
        else:
            # Calculate the mean & variance along self.axis (layer activations).
            mean, variance = tf.nn.moments(data, axes=self.axis, keepdims=True)
            gamma, beta = _broadcast(self.gamma), _broadcast(self.beta)

            inv = tf.math.rsqrt(variance + self.epsilon)
            if gamma is not None:
                gamma = tf.cast(gamma, data.dtype)
                inv = inv * gamma

            res = -mean * inv
            if beta is not None:
                beta = tf.cast(beta, data.dtype)
                res = res + beta

            outputs = data * inv + res

        return outputs
