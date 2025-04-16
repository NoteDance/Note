""" CAME
https://arxiv.org/abs/2307.02047

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class CAME(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=2e-4,
        beta1=0.9,
        beta2=0.999,
        beta3=0.9999,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        clip_threshold=1.0,
        ams_bound=False,
        eps1=1e-30,
        eps2=1e-16,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="came",
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
        self.beta3 = beta3
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.clip_threshold = clip_threshold
        self.ams_bound = ams_bound
        self.eps1 = eps1
        self.eps2 = eps2
    
    def reset(self):
        iterations = tf.Variable(
                0,
                name="iteration",
                dtype=tf.int64,
                trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
        self._track_variable(iterations)
        self._iterations = iterations
        for var in self._trainable_variables:
            grad_shape = var.shape
            factored = self.get_options(grad_shape)

            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            
            if factored:
                self.exp_avg_sq_row[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                        reference_variable=tf.Variable(tf.zeros(grad_shape[:-1], dtype=var.dtype)), name="exp_avg_sq_row"
                                    )
                self.exp_avg_sq_col[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                        reference_variable=tf.Variable(tf.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=var.dtype)), name="exp_avg_sq_col"
                                    )
                self.exp_avg_res_row[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                        reference_variable=tf.Variable(tf.zeros(grad_shape[:-1], dtype=var.dtype)), name="exp_avg_res_row"
                                    )
                self.exp_avg_res_col[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                        reference_variable=tf.Variable(tf.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=var.dtype)), name="exp_avg_res_col"
                                    )
            else:
                self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="exp_avg_sq"
                                                        )
            
            if self.ams_bound:
                self.exp_avg_sq_hat[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="exp_avg_sq_hat"
                                                        )
            
            self.RMS = 0.0
    
    @staticmethod
    def get_options(shape):
        r"""Get `factored`."""
        return len(shape) >= 2

    @staticmethod
    def get_rms(x):
        r"""Get RMS."""
        norm_val = tf.norm(x, ord=2)
        numel = tf.cast(tf.size(x), x.dtype)
        rms = norm_val / tf.sqrt(numel)
        return rms

    @staticmethod
    def approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, output):
        r"""Get approximation of EMA of squared gradient."""
        mean_row = tf.reduce_mean(exp_avg_sq_row, axis=-1, keepdims=True)
        
        r_factor = tf.math.rsqrt(exp_avg_sq_row / mean_row)
        r_factor = tf.expand_dims(r_factor, axis=-1)
        
        c_factor = tf.expand_dims(exp_avg_sq_col, axis=-2)
        c_factor = tf.math.rsqrt(c_factor)
        
        result = tf.multiply(r_factor, c_factor)
        output.assign(result)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg_sq_row = []
        self.exp_avg_sq_col = []
        self.exp_avg_res_row = []
        self.exp_avg_res_col = []
        if self.ams_bound:
            self.exp_avg_sq_hat = []
        for var in var_list:
            grad_shape = var.shape
            factored = self.get_options(grad_shape)
            
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            
            if factored:
                self.exp_avg_sq.append(tf.Variable(0))
                self._track_variable(self.exp_avg_sq[-1])
                self.exp_avg_sq_row.append(self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.zeros(grad_shape[:-1], dtype=var.dtype)), name="exp_avg_sq_row"
                                                        ))
                self.exp_avg_sq_col.append(self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=var.dtype)), name="exp_avg_sq_col"
                                                        ))
                self.exp_avg_res_row.append(self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.zeros(grad_shape[:-1], dtype=var.dtype)), name="exp_avg_res_row"
                                                        ))
                self.exp_avg_res_col.append(self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=var.dtype)), name="exp_avg_res_col"
                                                        ))
            else:
                self.exp_avg_sq_row.append(tf.Variable(0))
                self.exp_avg_sq_col.append(tf.Variable(0))
                self.exp_avg_res_row.append(tf.Variable(0))
                self.exp_avg_res_col.append(tf.Variable(0))
                self._track_variable(self.exp_avg_sq_row[-1])
                self._track_variable(self.exp_avg_sq_col[-1])
                self._track_variable(self.exp_avg_res_row[-1])
                self._track_variable(self.exp_avg_res_col[-1])
                self.exp_avg_sq.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        ))
            
            if self.ams_bound:
                self.exp_avg_sq_hat.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq_hat"
                                                        ))
            
            self.RMS = 0.0

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'CAME does not support sparse gradients')
        
        self.RMS = self.get_rms(variable)
        
        update = gradient * gradient + self.eps1
        
        grad_shape = variable.shape
        factored = self.get_options(grad_shape)
        
        if factored:
            exp_avg_sq_row = self.exp_avg_sq_row[self._get_variable_index(variable)]
            exp_avg_sq_col = self.exp_avg_sq_col[self._get_variable_index(variable)]

            exp_avg_sq_row.assign(exp_avg_sq_row * self.beta2 + tf.reduce_mean(update, axis=-1) * (1.0 - self.beta2))
            exp_avg_sq_col.assign(exp_avg_sq_col * self.beta2 + tf.reduce_mean(update, axis=-2) * (1.0 - self.beta2))

            self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
        else:
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
            exp_avg_sq.assign(exp_avg_sq * self.beta2 + update * (1.0 - self.beta2))
            update = tf.math.rsqrt(exp_avg_sq)
        
        if self.ams_bound:
            exp_avg_sq_hat = self.exp_avg_sq_hat[self._get_variable_index(variable)]
            exp_avg_sq_hat.assign(tf.maximum(exp_avg_sq_hat, 1.0 / update))
            update = tf.math.rsqrt(exp_avg_sq_hat / self.beta2)
        
        update = update * gradient
        
        update = update / tf.maximum(self.get_rms(update) / self.clip_threshold, 1.0)
        
        exp_avg = self.exp_avg [self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 + update * (1.0 - self.beta1))
        
        res = update - exp_avg
        res = tf.pow(res, 2) + self.eps2
        
        if factored:
            exp_avg_res_row = self.exp_avg_res_row[self._get_variable_index(variable)]
            exp_avg_res_col = self.exp_avg_res_col[self._get_variable_index(variable)]

            exp_avg_res_row.assign(exp_avg_res_row * self.beta3 + tf.reduce_mean(res, axis=-1) * (1.0 - self.beta3))
            exp_avg_res_col.assign(exp_avg_res_col * self.beta3 + tf.reduce_mean(res, axis=-2) * (1.0 - self.beta3))

            self.approximate_sq_grad(exp_avg_res_row, exp_avg_res_col, update)
            update = update * exp_avg
        else:
            update = exp_avg
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

        update = update * lr

        variable.assign_add(-update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "clip_threshold": self.clip_threshold,
                "ams_bound": self.ams_bound,
                "eps1": self.eps1,
                "eps2": self.eps2,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass