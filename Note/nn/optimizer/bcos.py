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


def unit_norm(x, ord = 2.0):
    r"""Get norm of unit."""
    keepdims = True
    axis = None

    x_len = len(x.shape)
    if x_len <= 1:
        keepdims = False
    elif x_len in (2, 3):
        axis = 1
    elif x_len == 4:
        axis = (1, 2, 3)
    else:
        axis = tuple(range(1, x_len))

    return tf.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def agc(
    p, grad, agc_eps = 1e-3, agc_clip_val = 1e-2, eps = 1e-6
):
    r"""Clip gradient values in excess of the unit wise norm."""
    max_norm = tf.maximum(unit_norm(p), agc_eps) * agc_clip_val
    g_norm = tf.maximum(unit_norm(grad), eps)

    clipped_grad = grad * (max_norm / g_norm)

    return tf.where(g_norm > max_norm, clipped_grad, grad)


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
        agc=False,
        gc=False,
        sophia=False,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        trust_ratio=False,
        trust_clip=False,
        cautious=False,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=False,
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
        self.agc = agc
        self.gc = gc
        self.sophia = sophia
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.cautious = cautious
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        
        self.m = []
        self.v = []
        self.subset_size_ = []
        self.hessian_moment = []
        self.hessian = []
        self.slow_momentum = []
        
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
                    if self.sophia:
                        self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                                    reference_variable=second_moment_update, name="hessian"
                                                                )
                        self.hessian_moment.append(self.add_variable_from_reference(
                            reference_variable=second_moment_update, name="hessian_moment"
                                                ))
                    else:
                        self.v.append(self.add_variable_from_reference(
                                reference_variable=second_moment_update, name="variance"
                            ))
                else:
                    if self.sophia:
                        self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                                    reference_variable=var, name="hessian"
                                                                )
                        self.hessian_moment.append(self.add_variable_from_reference(
                            reference_variable=var, name="hessian_moment"
                                                ))
                    else:
                        self.v.append(self.add_variable_from_reference(
                                reference_variable=var, name="variance"
                            ))
            else:
                self.v.append(None)
            
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])

    def compute_v(self, gradient, m, beta, beta2, idx, step):
        if self.sophia:
            hessian_moment = self.hessian_moment[idx]
            def true_fn2():
                hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[idx] * (1.0 - self.beta2))
            def false_fn2():
                pass
            tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
            g2 = hessian_moment
        else:
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
        
    def compute_hutchinson_hessian(
        self,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params = [p for p in self._trainable_variables if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [g for g in grads if not tf.keras.backend.is_sparse(g)]

        for i in range(num_samples):
            if distribution == 'rademacher':
                zs = [
                    tf.cast(tf.random.uniform(tf.shape(p), 0, 2, dtype=tf.int32)*2 - 1, p.dtype)
                    for p in params
                ]
            else:
                zs = [tf.random.normal(tf.shape(p), dtype=p.dtype) for p in params]

            h_zs = self.tape.gradient(grads, params, zs)

            for h_z, z, p in zip(h_zs, zs, params):
                size = tf.size(p)
                if self.sn:
                    reshaped_h_z = tf.reshape(h_z, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    reshaped_z = tf.reshape(z, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    hessian_update = tf.reduce_sum(reshaped_h_z * reshaped_z, axis=1, keepdims=True)
                else:
                    hessian_update = h_z * z
                self.hessian[self._get_variable_index(p)].assign_add(hessian_update * alpha / num_samples)
    
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

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.sophia:
            def true_fn1():
                self.compute_hutchinson_hessian(
                    grads,
                    num_samples=self.num_samples,
                    distribution=self.distribution,
                )
            def false_fn1():
                pass
            tf.cond(self.iterations % self.update_period == 0, true_fn1, false_fn1)
            
        for variable, gradient in zip(trainable_variables, grads):
            step = tf.cast(self.iterations + 1, variable.dtype)
            lr = tf.cast(learning_rate, variable.dtype)
            
            idx = self._get_variable_index(variable)
            
            if self.agc:
                grads[idx] = agc(variable, gradient)
                gradient = grads[idx]
            
            if self.gc:
                size = len(gradient.shape)
                if size > 1:
                    grads[idx] += tf.reduce_mean(-gradient, axis=tuple(range(1, size)), keepdims=True)
                def true_fn():
                    s = tf.math.reduce_std(grads[idx]) + 1e-8
                    grads[idx] = grads[idx] / s
                def false_fn():
                    pass
                tf.cond(tf.size(gradient) > 2, true_fn, false_fn)
                gradient = grads[idx]
            
            size = tf.size(gradient)
            
            if self.maximize:
                gradient = -gradient
            
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
                if self.sophia:
                    hessian_moment = self.hessian_moment[idx]
                    def true_fn2():
                        hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[idx] * (1.0 - self.beta2))
                    def false_fn2():
                        pass
                    tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
                else:
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
                v = self.compute_v(gradient, old_m, self.beta, self.beta2, idx, step)
            
            if self.sn:
                d = tf.reshape(d, (size // self.subset_size_[idx], self.subset_size_[idx]))
                if self.sophia:
                    de_nom = tf.maximum(hessian_moment, self.epsilon)
                    update = tf.reshape(d / de_nom, variable.shape)
                    update = tf.clip_by_value(update, clip_value_min=-self.p, clip_value_max=self.p)
                else:
                    update = tf.reshape(d / (tf.sqrt(v) + self.epsilon), variable.shape)
            else:
                if self.sophia:
                    de_nom = tf.maximum(hessian_moment, self.epsilon)
                    update = tf.clip_by_value(d / de_nom, clip_value_min=-self.p, clip_value_max=self.p)
                else:
                    update = d / (tf.sqrt(v) + self.epsilon)
            
            if self.trust_ratio:
                # Layer-wise LR adaptation
                w_norm = tf.norm(variable, ord=2)
                g_norm = tf.norm(update, ord=2)
                trust_ratio = w_norm / g_norm
                trust_ratio = tf.where(
                    w_norm > 0,
                    tf.where(g_norm > 0, trust_ratio, 1.0),
                    1.0,
                )
                if self.trust_clip:
                    trust_ratio = tf.minimum(trust_ratio, 1.0)
                update *= trust_ratio
                
            if self.cautious:
                mask = tf.cast(tf.math.greater(update * gradient, 0), gradient.dtype)
                numel = tf.cast(tf.size(mask), gradient.dtype)
                factor = numel / (tf.reduce_sum(mask) + 1)
                mask = mask * factor
                update = update * mask
            
            variable.assign_add(-lr * update) 
            
            if self.lookahead:
                def true_fn():
                    slow_p = self.slow_momentum[idx]
                    slow_p.assign(slow_p + self.lookahead_blending_alpha * (variable - slow_p))
                    variable.assign(slow_p)
                
                def false_fn():
                    pass
            
                tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)

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
                "agc": self.agc,
                "gc": self.gc,
                "sophia": self.sophia,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
                "cautious": self.cautious,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "maximize": self.maximize,
            }
        )
        return config
    
    def _apply_weight_decay(self, variables):
        pass
