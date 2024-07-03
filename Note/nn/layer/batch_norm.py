import tensorflow as tf
from keras import backend
from keras import ops
from Note.nn.initializer import initializer
from multiprocessing import Manager
from Note.nn.Model import Model


class batch_norm:
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', synchronized=False, trainable=True, dtype='float32'):
        self.input_size=input_size
        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.moving_mean_initializer=moving_mean_initializer
        self.moving_variance_initializer=moving_variance_initializer
        self.synchronized=synchronized
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=input_size
        self.train_flag=True
        if input_size!=None:
            self.moving_mean=initializer([input_size], moving_mean_initializer, dtype)
            self.moving_variance=initializer([input_size], moving_variance_initializer, dtype)
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
            Model.param.extend(self.param)
        Model.layer_list.append(self)
        if Model.name_!=None and Model.name_ not in Model.layer_eval:
            Model.layer_eval[Model.name_]=[]
            Model.layer_eval[Model.name_].append(self)
        elif Model.name_!=None:
            Model.layer_eval[Model.name_].append(self)
    
    
    def build(self):
        self.output_size=self.input_size
        self.moving_mean=initializer([self.input_size], self.moving_mean_initializer, self.dtype)
        self.moving_variance=initializer([self.input_size], self.moving_variance_initializer, self.dtype)
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
        Model.param.extend(self.param)
        return
    
    
    def __call__(self, data, training=None, mask=None):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if training==None:
            training=self.train_flag
        if training and self.trainable:
            mean, variance = self._moments(
                data,
                mask,
            )
            moving_mean = ops.cast(self.moving_mean, data.dtype)
            moving_variance = ops.cast(self.moving_variance, data.dtype)
            self.moving_mean.assign(
                ops.cast(
                    moving_mean * self.momentum + mean * (1.0 - self.momentum),
                    data.dtype,
                )
            )
            self.moving_variance.assign(
                ops.cast(
                    moving_variance * self.momentum
                    + variance * (1.0 - self.momentum),
                    data.dtype,
                )
            )
        else:
            moving_mean = ops.cast(self.moving_mean, data.dtype)
            moving_variance = ops.cast(self.moving_variance, data.dtype)
            mean = moving_mean
            variance = moving_variance
    
        if self.scale:
            gamma = ops.cast(self.gamma, data.dtype)
        else:
            gamma = None
    
        if self.center:
            beta = ops.cast(self.beta, data.dtype)
        else:
            beta = None
    
        outputs = ops.batch_normalization(
            x=data,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )
        return outputs
    
    
    def _moments(self, inputs, mask):
        reduction_axes = list(range(len(inputs.shape)))
        del reduction_axes[self.axis]
        _reduction_axes = reduction_axes
        if mask is None:
            return ops.moments(
                inputs,
                axes=_reduction_axes,
                synchronized=self.synchronized,
            )

        mask_weights = ops.cast(
            mask,
            inputs.dtype,
        )
        mask_weights_broadcasted = ops.expand_dims(
            mask_weights,
            axis=-1,
        )
        weighted_inputs = mask_weights_broadcasted * inputs

        weighted_input_sum = ops.sum(
            weighted_inputs,
            _reduction_axes,
            keepdims=True,
        )
        sum_of_weights = ops.sum(
            mask_weights_broadcasted,
            _reduction_axes,
            keepdims=True,
        )
        mean = weighted_input_sum / (sum_of_weights + backend.config.epsilon())

        difference = weighted_inputs - mean
        squared_difference = ops.square(difference)
        weighted_distsq = ops.sum(
            mask_weights_broadcasted * squared_difference,
            _reduction_axes,
            keepdims=True,
        )
        variance = weighted_distsq / (sum_of_weights + backend.config.epsilon())

        return ops.squeeze(mean), ops.squeeze(variance)


class batch_norm_:
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', keepdims=True, trainable=True, parallel=True, dtype='float32'):
        self.input_size=input_size
        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.moving_mean_initializer=moving_mean_initializer
        self.moving_variance_initializer=moving_variance_initializer
        self.keepdims=keepdims
        self.trainable=trainable
        self.parallel=parallel
        self.dtype=dtype
        self.output_size=input_size
        self.train_flag=True
        if input_size!=None:
            self.moving_mean=initializer([input_size], moving_mean_initializer, dtype)
            self.moving_variance=initializer([input_size], moving_variance_initializer, dtype)
            if parallel:
                manager=Manager()
                self.moving_mean=manager.list([self.moving_mean])
                self.moving_var=manager.list([self.moving_var])
                Model.ctl_list.append(self.convert_to_list)
                Model.ctsl_list.append(self.convert_to_shared_list)
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
            Model.param.extend(self.param)
        Model.layer_list.append(self)
        if Model.name_!=None and Model.name_ not in Model.layer_eval:
            Model.layer_eval[Model.name_]=[]
            Model.layer_eval[Model.name_].append(self)
        elif Model.name_!=None:
            Model.layer_eval[Model.name_].append(self)
    
    
    def build(self):
        self.output_size=self.input_size
        self.moving_mean=initializer([self.input_size], self.moving_mean_initializer, self.dtype)
        self.moving_variance=initializer([self.input_size], self.moving_variance_initializer, self.dtype)
        if self.parallel:
            manager=Manager()
            self.moving_mean=manager.list([self.moving_mean])
            self.moving_var=manager.list([self.moving_var])
            Model.ctl_list.append(self.convert_to_list)
            Model.ctsl_list.append(self.convert_to_shared_list)
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
        Model.param.extend(self.param)
        return
    
    
    def __call__(self, data, training=None):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if training==None:
            training=self.train_flag
        if training:
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
