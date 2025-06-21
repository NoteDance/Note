""" AdamWSN
https://arxiv.org/abs/2411.07120

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def closest_smaller_divisor_of_n_to_k(n, k):
    r"""Get closest smaller divisor of n to k."""
    def true_fn():
        return k
    
    def false_fn():
        def true_fn():
            raise ValueError
        def false_fn():
            pass
        tf.cond(tf.logical_or(n <= 1, k <= 1), true_fn, false_fn)
        closest_smaller_divisor = -2
        for i in tf.range(k, 0, -1):
            def true_fn():
                def true_fn():
                    return i
                def false_fn():
                    return closest_smaller_divisor
                return tf.cond(closest_smaller_divisor == -2, true_fn, false_fn)
            def false_fn():
                return -1  # pragma: no cover
            closest_smaller_divisor = tf.cond(n % i == 0, true_fn, false_fn)
        return closest_smaller_divisor
    
    closest_smaller_divisor = tf.cond(n % k == 0, true_fn, false_fn)
    
    return closest_smaller_divisor


class AdamWSN(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        subset_size=-1,
        sn=True,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamwsn",
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
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.subset_size = subset_size
        self.sn = sn
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.subset_size_ = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                ))
            if self.sn:
                size = tf.size(var)
                
                def true_fn():
                    return self.subset_size
                def false_fn():
                    return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                    size,
                    tf.cond(self.subset_size > 0, true_fn, false_fn)
                ))

                reshaped_grad = tf.reshape(var, (size // self.subset_size_[-1], self.subset_size_[-1]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
                second_moment_update = tf.Variable(second_moment_update)
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="exp_avg_sq"
                    ))
            else:
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg_sq"
                    ))

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdamWSN does not support sparse gradient.')
        
        if variable.dtype.is_complex:
            raise RuntimeError(
                'AdamWSN does not support complex parameter.')
        
        step = tf.cast(self.iterations + 1, variable.dtype)
            
        lr = tf.cast(learning_rate, variable.dtype)
        
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
        
        step_size = lr * bias_correction2_sq / bias_correction1
        
        size = tf.size(gradient)
        
        if self.maximize:
            gradient = -gradient
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        if self.sn:
            reshaped_grad = tf.reshape(gradient, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
            second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
        else:
            second_moment_update = tf.pow(gradient, 2)
        
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + second_moment_update * (1.0 - self.beta2))
        
        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        if self.sn:
            numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
            norm_grad = tf.reshape((numerator / de_nom), variable.shape)
            variable.assign_add(norm_grad * -step_size)
        else:
            variable.assign_add(-step_size * exp_avg / de_nom)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "maximize": self.maximize,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass