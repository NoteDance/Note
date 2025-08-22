""" SophiaH
https://arxiv.org/abs/2305.14342

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


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


class SophiaH(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=6e-2,
        beta1=0.96,
        beta2=0.99,
        epsilon=1e-12,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sophiah",
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
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum"
                                                    )
            self.hessian_moment[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="hessian_moment"
                                                    )
            self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="hessian"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum = []
        self.hessian_moment = []
        self.hessian = []
        for var in var_list:
            self.momentum.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum"
                                                    ))
            self.hessian_moment.append(self.add_variable_from_reference(
                                reference_variable=var, name="hessian_moment"
                                                    ))
            self.hessian.append(self.add_variable_from_reference(
                                reference_variable=var, name="hessian"
                                                    ))
    
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
                self.hessian[self._get_variable_index(p)].assign_add(h_z * z * alpha / num_samples)

    def update_step(self, grads, trainable_variables, learning_rate):
        def true_fn1():
            self.compute_hutchinson_hessian(
                grads,
                num_samples=self.num_samples,
                distribution=self.distribution,
            )
        def false_fn1():
            pass
        tf.cond(self.iterations % self.update_period == 0, true_fn1, false_fn1)
        
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'SophiaH does not support sparse gradients')
            
            lr = tf.cast(learning_rate, p.dtype)
            
            step = tf.cast(self.iterations + 1, p.dtype)

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * self.weight_decay

            momentum = self.momentum[self._get_variable_index(p)]
            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
            momentum.assign(momentum * self.beta1 + g * (1.0 - self.beta1))
            
            def true_fn2():
                hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))
            def false_fn2():
                pass
            tf.cond(step % self.update_period == 0, true_fn2, false_fn2)

            update = tf.clip_by_value(momentum / tf.maximum(hessian_moment, self.epsilon), clip_value_min=-p, clip_value_max=p)
            p.assign_add(update * -lr)       

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class SophiaH_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=6e-2,
        beta1=0.96,
        beta2=0.99,
        epsilon=1e-12,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        orthograd=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        subset_size=-1,
        sn=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sophiah_e",
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
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.orthograd = orthograd
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.subset_size = subset_size
        self.sn = sn
    
    def reset(self):
        self._iterations.assign(0)
        self.momentum = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        for var in self._trainable_variables:
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])
            if self.pnm:
                self.pos_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="pos_momentum"
                    )
                )
                self.neg_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="neg_momentum"
                    )
                )
            else:
                self.momentum.append(self.add_variable_from_reference(
                                    reference_variable=var, name="momentum"
                                                        ))
            self.hessian_moment[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="hessian_moment"
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
                self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=second_moment_update, name="hessian"
                                                        )
            else:
                self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="hessian"
                                                        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum = []
        self.hessian_moment = []
        self.hessian = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        for var in var_list:
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])
            if self.pnm:
                self.pos_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="pos_momentum"
                    )
                )
                self.neg_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="neg_momentum"
                    )
                )
            else:
                self.momentum.append(self.add_variable_from_reference(
                                    reference_variable=var, name="momentum"
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
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                second_moment_update = tf.Variable(second_moment_update)
                self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=second_moment_update, name="hessian"
                                                        )
                self.hessian_moment.append(self.add_variable_from_reference(
                    reference_variable=second_moment_update, name="hessian_moment"
                                        ))
            else:
                self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="hessian"
                                                        )
                self.hessian_moment.append(self.add_variable_from_reference(
                    reference_variable=var, name="hessian_moment"
                                        ))
    
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
    
    def apply_orthogonal_gradients(self, params, grads, eps = 1e-16):
        for p, g in zip(params, grads):
            if tf.keras.backend.is_sparse(g):
                continue
            
            original_shape = g.shape
            w = tf.reshape(p, [-1])
            g = tf.reshape(g, [-1])
    
            proj = tf.tensordot(w, g, axes=1) / (tf.tensordot(w, w, axes=1) + eps)
            g_ortho = tf.cast(g, tf.float32) - proj * w
            g_norm = tf.norm(g)
            g_ortho_norm = tf.norm(g_ortho)
            g_ortho_scaled = g_ortho * (g_norm / (g_ortho_norm + eps))
            
            grads[self._get_variable_index(p)] = tf.reshape(g_ortho_scaled, original_shape)
    
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

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)
        def true_fn1():
            self.compute_hutchinson_hessian(
                grads,
                num_samples=self.num_samples,
                distribution=self.distribution,
            )
        def false_fn1():
            pass
        tf.cond(self.iterations % self.update_period == 0, true_fn1, false_fn1)
        
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'SophiaH_e does not support sparse gradients')
            
            lr = tf.cast(learning_rate, p.dtype)
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            size = tf.size(p)

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * self.weight_decay

            momentum = self.momentum[self._get_variable_index(p)]
            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
            if not self.pnm:
                momentum.assign(momentum * self.beta1 + g * (1.0 - self.beta1))
            else:
                noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                def true_fn():
                    return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                def false_fn():
                    return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                pos_momentum.assign(pos_momentum * self.beta1 ** 2 + g * (1.0 - self.beta1 ** 2))
                momentum = pos_momentum * (1 + self.beta2) + neg_momentum * -self.beta2 * (1.0 / noise_norm)
            
            def true_fn2():
                hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))
            def false_fn2():
                pass
            tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
            
            if self.sn:
                numerator = tf.reshape(momentum, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                norm_grad = tf.reshape((numerator / tf.maximum(hessian_moment, self.epsilon)), p.shape)
                update = tf.clip_by_value(norm_grad, clip_value_min=-p, clip_value_max=p)
            else:
                update = tf.clip_by_value(momentum / tf.maximum(hessian_moment, self.epsilon), clip_value_min=-p, clip_value_max=p)

            p.assign_add(update * -lr) 
            
            if self.lookahead:
                def true_fn():
                    slow_p = self.slow_momentum[self._get_variable_index(p)]
                    slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                    p.assign(slow_p)
                
                def false_fn():
                    pass
            
                tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
                "orthograd": self.orthograd,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
