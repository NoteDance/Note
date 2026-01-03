import tensorflow as tf
from keras.src.optimizers import optimizer
from typing import Literal


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


Mode = Literal['g', 'm', 'c']


class BCOS(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.9,
        beta2=None,
        mode='c',
        simple_cond=False,
        weight_decay=0.1,
        weight_decouple=True,
        epsilon=1e-6,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="bcos",
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
        self.beta = beta
        self.beta2 = beta2
        self.mode = mode
        self.simple_cond = simple_cond
        self.epsilon = epsilon
        self.weight_decouple = weight_decouple
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        
        self.m = []
        self.v = []
        
        for var in var_list:
            if self.mode in ('m', 'c'):
                self.m.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                ))
            else:
                self.m.append(None)
            
            if self.mode in ('g', 'm'):
                self.v.append(self.add_variable_from_reference(
                    reference_variable=var, name="variance"
                ))
            else:
                self.v.append(None)

    def compute_v(self, gradient, m, beta, beta2):
        g2 = gradient * gradient
        
        if self.simple_cond:
            beta_v = 1.0 - (1.0 - beta) ** 2 if beta2 is None else beta2
            return beta_v * m * m + (1.0 - beta_v) * g2
        
        return (
            (3.0 * beta ** 2 - 2.0 * beta ** 3) * m * m
            + (1.0 - beta) ** 2 * g2
            + 2.0 * beta * (1.0 - beta) ** 2 * m * gradient
        )

    def update_step(self, gradient, variable, learning_rate):
        step = tf.cast(self.iterations + 1, variable.dtype)
        lr = tf.cast(learning_rate, variable.dtype)
        
        if self.maximize:
            gradient = -gradient
        
        idx = self._get_variable_index(variable)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * lr))
        elif self.weight_decay > 0.0:
            gradient = gradient + variable * self.weight_decay
        
        if self.mode in ('m', 'c'):
            m = self.m[idx]
            def true_fn():
                m.assign(gradient)
            def false_fn():
                pass
            tf.cond(step == 1, true_fn, false_fn)
            old_m = tf.identity(m)
        
        if self.mode in ('m', 'c'):
            m = self.m[idx]
            m.assign(m * self.beta + gradient * (1.0 - self.beta))
            d = m
        else:
            d = gradient
        
        if self.mode in ('g', 'm'):
            beta_v = self.beta if self.beta2 is None else self.beta2
            v = self.v[idx]
            def true_fn():
                v.assign(d * d)
            def false_fn():
                v.assign(v * beta_v + d * d * (1.0 - beta_v))
            tf.cond(step == 1, true_fn, false_fn)
        else:
            v = self.compute_v(gradient, old_m, self.beta, self.beta2)
        
        variable.assign_add(-lr * d / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "beta2": self.beta2,
                "mode": self.mode,
                "simple_cond": self.simple_cond,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "maximize": self.maximize,
            }
        )
        return config
    
    def _apply_weight_decay(self, variables):
        pass


class BCOS_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.9,
        beta2=None,
        mode='c',
        simple_cond=False,
        weight_decay=0.1,
        weight_decouple=True,
        epsilon=1e-6,
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
        name="bcos_e",
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
        self.beta = beta
        self.beta2 = beta2
        self.mode = mode
        self.simple_cond = simple_cond
        self.epsilon = epsilon
        self.weight_decouple = weight_decouple
        self.subset_size = subset_size
        self.sn = sn
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        
        self.m = []
        self.v = []
        self.subset_size_ = []
        
        for var in var_list:
            if self.mode in ('m', 'c'):
                self.m.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                ))
            else:
                self.m.append(None)
            
            if self.mode in ('g', 'm'):
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
                    self.v.append(self.add_variable_from_reference(
                            reference_variable=second_moment_update, name="variance"
                        ))
                else:
                    self.v.append(self.add_variable_from_reference(
                            reference_variable=var, name="variance"
                        ))
            else:
                self.v.append(None)

    def compute_v(self, gradient, m, beta, beta2, idx):
        size = tf.size(gradient)
        if self.sn:
            reshaped_grad = tf.reshape(gradient, (size // self.subset_size_[idx], self.subset_size_[idx]))
            second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
        else:
            second_moment_update = tf.pow(gradient, 2)
        g2 = second_moment_update
        
        if self.sn:
            m = tf.reshape(m, (size // self.subset_size_[idx], self.subset_size_[idx]))
        
        if self.simple_cond:
            beta_v = 1.0 - (1.0 - beta) ** 2 if beta2 is None else beta2
            return beta_v * m * m + (1.0 - beta_v) * g2
        
        if self.sn:
            return (
                (3.0 * beta ** 2 - 2.0 * beta ** 3) * m * m
                + (1.0 - beta) ** 2 * g2
                + 2.0 * beta * (1.0 - beta) ** 2 * m * reshaped_grad
            )
        else:
            return (
                (3.0 * beta ** 2 - 2.0 * beta ** 3) * m * m
                + (1.0 - beta) ** 2 * g2
                + 2.0 * beta * (1.0 - beta) ** 2 * m * gradient
            )

    def update_step(self, gradient, variable, learning_rate):
        step = tf.cast(self.iterations + 1, variable.dtype)
        lr = tf.cast(learning_rate, variable.dtype)
        
        size = tf.size(gradient)
        
        if self.maximize:
            gradient = -gradient
        
        idx = self._get_variable_index(variable)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * lr))
        elif self.weight_decay > 0.0:
            gradient = gradient + variable * self.weight_decay
        
        if self.mode in ('m', 'c'):
            m = self.m[idx]
            def true_fn():
                m.assign(gradient)
            def false_fn():
                pass
            tf.cond(step == 1, true_fn, false_fn)
            old_m = tf.identity(m)
        
        if self.mode in ('m', 'c'):
            m = self.m[idx]
            m.assign(m * self.beta + gradient * (1.0 - self.beta))
            d = m
        else:
            d = gradient
        
        if self.sn:
            d = tf.reshape(d, (size // self.subset_size_[idx], self.subset_size_[idx]))
        
        if self.mode in ('g', 'm'):
            beta_v = self.beta if self.beta2 is None else self.beta2
            v = self.v[idx]
            if self.sn:
                reshaped_grad = tf.reshape(d, (size // self.subset_size_[idx], self.subset_size_[idx]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
            else:
                second_moment_update = tf.pow(d, 2)
            def true_fn():
                v.assign(second_moment_update)
            def false_fn():
                v.assign(v * beta_v + second_moment_update * (1.0 - beta_v))
            tf.cond(step == 1, true_fn, false_fn)
        else:
            v = self.compute_v(gradient, old_m, self.beta, self.beta2, idx)
        
        if self.sn:
            norm_grad = tf.reshape(d / (tf.sqrt(v) + self.epsilon), variable.shape)
            variable.assign_add(-lr * norm_grad)
        else:
            variable.assign_add(-lr * d / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "beta2": self.beta2,
                "mode": self.mode,
                "simple_cond": self.simple_cond,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "maximize": self.maximize,
            }
        )
        return config
    
    def _apply_weight_decay(self, variables):
        pass
