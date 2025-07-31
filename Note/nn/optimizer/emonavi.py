from typing import Dict, Union

import tensorflow as tf
from keras.src.optimizers import optimizer


def update_ema(ema: Dict, loss: Union[float, tf.Tensor]):
    r"""Update the EMA dictionary for the `short` and `long` terms."""
    ema['short'] = 0.3 * loss + 0.7 * ema.get('short', loss)
    ema['long'] = 0.01 * loss + 0.99 * ema.get('long', loss)

    return


def compute_scalar(ema: Dict[str, float]):
    r"""Compute the difference scalar."""
    diff = ema['short'] - ema['long']
    return tf.math.tanh(5.0 * diff) 


def get_scalar_ratio(scalar):
    r"""Get the scalar ratio."""
    if scalar > 0.6:
        return 0.7 + 0.2 * scalar
    if scalar < -0.6:
        return 0.1
    if tf.abs(scalar) > 0.3:
        return 0.3
    return 0.0


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
        closest_smaller_divisor = -7
        for i in tf.range(k, 0, -1):
            def true_fn():
                def true_fn():
                    return i
                def false_fn():
                    return -7
                return tf.cond(closest_smaller_divisor == -7, true_fn, false_fn)
            def false_fn():
                return -7  # pragma: no cover
            closest_smaller_divisor = tf.cond(n % i == 0, true_fn, false_fn)
        return closest_smaller_divisor
    
    closest_smaller_divisor = tf.cond(n % k == 0, true_fn, false_fn)
    
    def true_fn():
        return -1
    def false_fn():
        return closest_smaller_divisor
    closest_smaller_divisor = tf.cond(closest_smaller_divisor == -7, true_fn, false_fn)
    
    return closest_smaller_divisor


class EmoNavi(optimizer.Optimizer):
    r"""An emotion-driven optimizer that feels loss and navigates accordingly.

    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param shadow_weight: float. the weight of the shadow.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        learning_rate = 1e-3,
        betas = (0.9, 0.999),
        epsilon = 1e-8,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        shadow_weight: float = 0.05,
        maximize: bool = False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name = "emonavi",
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
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.shadow_weight = shadow_weight
        self.maximize = maximize
        self.ema = {}
    
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.shadow = []
        self.exp_avg = []
        self.exp_avg_sq = []
        for var in var_list:
            self.shadow.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="shadow"
                )
            )
            self.shadow[-1].assign(var)
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
    
    def apply_gradients(self, grads_and_vars, loss):
        self.loss = loss
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'EmoNavi does not support sparse gradients')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        beta1, beta2 = self.betas

        if self.maximize:
            gradient = -gradient

        update_ema(self.ema, self.loss)
        scalar = compute_scalar(self.ema)
        ratio = get_scalar_ratio(scalar)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

        shadow = self.shadow[self._get_variable_index(variable)]
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        def true_fn():
            variable.assign(variable + ratio * (shadow - variable))
            shadow.assign(shadow + self.shadow_weight * (variable - shadow))
        
        def false_fn():
            pass
        
        tf.cond(ratio > 0.0, true_fn, false_fn)

        exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
        exp_avg_sq.assign(exp_avg_sq * beta2 + gradient * gradient * (1.0 - beta2))

        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        variable.assign_add(-lr * exp_avg / de_nom)
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "shadow_weight": self.shadow_weight,
                "maximize": self.maximize,
                "ema": self.ema,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class EmoNavi_sn(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate = 1e-3,
        betas = (0.9, 0.999),
        epsilon = 1e-8,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        shadow_weight: float = 0.05,
        subset_size = -1,
        sn = True,
        maximize: bool = False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name = "emonavi_sn",
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
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.shadow_weight = shadow_weight
        self.subset_size = subset_size
        self.sn = sn
        self.maximize = maximize
        self.ema = {}
    
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.shadow = []
        self.exp_avg = []
        self.exp_avg_sq = []
        self.subset_size_ = []
        for var in var_list:
            self.shadow.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="shadow"
                )
            )
            self.shadow[-1].assign(var)
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
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
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                second_moment_update = tf.Variable(second_moment_update)
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="exp_avg_sq"
                    ))
            else:
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg_sq"
                    ))
    
    def apply_gradients(self, grads_and_vars, loss):
        self.loss = loss
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'EmoNavi does not support sparse gradients')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        size = tf.size(gradient)
        
        beta1, beta2 = self.betas

        if self.maximize:
            gradient = -gradient

        update_ema(self.ema, self.loss)
        scalar = compute_scalar(self.ema)
        ratio = get_scalar_ratio(scalar)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

        shadow = self.shadow[self._get_variable_index(variable)]
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        def true_fn():
            variable.assign(variable + ratio * (shadow - variable))
            shadow.assign(shadow + self.shadow_weight * (variable - shadow))
        
        def false_fn():
            pass
        
        tf.cond(ratio > 0.0, true_fn, false_fn)
        
        if self.sn:
            reshaped_grad = tf.reshape(gradient, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
            second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
        else:
            second_moment_update = tf.pow(gradient, 2)

        exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
        exp_avg_sq.assign(exp_avg_sq * beta2 + second_moment_update * (1.0 - beta2))
        
        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        if self.sn:
            numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
            normed_grad = tf.reshape((numerator / de_nom), variable.shape)
            variable.assign_add(-lr * normed_grad)
        else:
            variable.assign_add(-lr * exp_avg / de_nom)
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "shadow_weight": self.shadow_weight,
                "sn": self.sn,
                "maximize": self.maximize,
                "ema": self.ema,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass