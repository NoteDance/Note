""" AdaHessian Optimizer

Lifted from https://github.com/davda54/ada-hessian/blob/master/ada_hessian.py
Originally licensed MIT, Copyright 2020, David Samuel
Modifications Copyright 2024 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
    

class Adahessian(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        hessian_power=1.0,
        update_each=1,
        n_samples=1,
        avg_conv_kernel=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adahessian",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.hessian_power = hessian_power
        self.update_each = update_each
        self.n_samples = n_samples
        self.avg_conv_kernel = avg_conv_kernel
        
        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.seed = 2147483647
        self.generator = tf.random.Generator.from_seed(self.seed)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_hessian_diag_sq = []
        self.hessian_step = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_hessian_diag_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_hessian_diag_sq"
                )
            )
            var.hess = 0.0
            self.hessian_step.append(tf.Variable(0, dtype=tf.int64))
            self._track_variable(self.hessian_step[-1])
    
    def apply_gradients(self, grads_and_vars, tape):
        self.tape = tape
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)
    
    @property
    def is_second_order(self):
        return True
    
    def zero_hessian(self, trainable_variables):
        """
        Zeros out the accumalated hessian traces.
        """

        for i,p in enumerate(trainable_variables):
            if not isinstance(p.hess, float) and tf.get_static_value(self.hessian_step[i]) % self.update_each == 0:
                p.hess = tf.zeros(p.shape)
    
    def set_hessian(self, grads, trainable_variables):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for i,p in enumerate(trainable_variables):
            if tf.get_static_value(self.hessian_step[i]) % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.hessian_step[i] += 1
        
        if len(params) == 0:
            return

        for i in range(self.n_samples):
            # Rademacher distribution {-1.0, 1.0}
            zs = [tf.cast(self.generator.uniform(shape=p.shape, minval=0, maxval=2, dtype=tf.int32), p.dtype) * 2.0 - 1.0 for p in params]
            h_zs = self.tape.gradient(grads, params, output_gradients=zs)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        self.zero_hessian(trainable_variables)
        self.set_hessian(grads, trainable_variables)
        
        for p, grad in zip(trainable_variables, grads):
            if self.avg_conv_kernel and len(p.shape) == 4:
                p.hess = tf.broadcast_to(tf.reduce_mean(tf.abs(p.hess), axis=[1, 2], keepdims=True), p.hess.shape)
        
            # Perform correct stepweight decay as in AdamW
            p.assign(p * (1 - lr * self.weight_decay))
            
            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_hessian_diag_sq = self.exp_hessian_diag_sq[self._get_variable_index(p)]
            step = tf.get_static_value(self.iterations + 1)
            
            # Decay the first and second moment running average coefficient
            exp_avg.assign(self.beta1 * exp_avg + (1 - self.beta1) * grad)
            exp_hessian_diag_sq.assign(self.beta2 * exp_hessian_diag_sq + (1 - self.beta2) * tf.square(p.hess))
           
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            
            k = self.hessian_power
            denom = tf.pow(exp_hessian_diag_sq / bias_correction2, k / 2) + self.epsilon
            
            # Make update
            step_size = lr / bias_correction1
            p.assign_add(-step_size * exp_avg / denom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "hessian_power": self.hessian_power,
                "update_each": self.update_each,
                "n_samples": self.n_samples,
                "avg_conv_kernel": self.avg_conv_kernel,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass	