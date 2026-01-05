""" Sophia
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
                p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * tf.cast(self.weight_decay, p.dtype)

            momentum = self.momentum[self._get_variable_index(p)]
            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
            momentum.assign(momentum * self.beta1 + g * (1.0 - self.beta1))
            
            def true_fn2():
                hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))
            def false_fn2():
                pass
            tf.cond(step % self.update_period == 0, true_fn2, false_fn2)

            update = tf.clip_by_value(momentum / tf.maximum(hessian_moment, self.epsilon), clip_value_min=-self.p, clip_value_max=self.p)
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
        orthograd=False,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=False,
        pnm=False,
        subset_size=-1,
        sn=False,
        agc=False,
        cautious=False,
        d0=1e-6,
        growth_rate=float('inf'),
        DAdapt=False,
        trust_ratio=False,
        trust_clip=False,
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
        self.lr = learning_rate
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
        self.agc = agc
        self.cautious = cautious
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip

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
        if self.DAdapt:
            self.s = []
            self.sk_l1 = tf.Variable(0.0)
            self.numerator_acc = tf.Variable(0.0)
            self.numerator_weighted = tf.Variable(0.0)
            self.d0_ = tf.Variable(self.d0)
            self._track_variable(self.sk_l1)
            self._track_variable(self.numerator_acc)
            self._track_variable(self.numerator_weighted)
            self._track_variable(self.d0_)
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
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                    reference_variable=var, name="s"
                                        ))
    
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
            if self.DAdapt:
                d_lr = self.d0 * lr
                d_lr = tf.cast(d_lr, p.dtype)
                s = self.s[self._get_variable_index(p)]
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            size = tf.size(p)
                         
            if self.agc:
                grads[self._get_variable_index(p)] = agc(p, g)
                g = grads[self._get_variable_index(p)]

            if not self.pnm:
                momentum = self.momentum[self._get_variable_index(p)]
            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
            if not self.pnm:
                if self.DAdapt:
                    beta2_sq = math.sqrt(self.beta2)
                    momentum.assign(momentum * self.beta1 + g * d_lr * (1.0 - self.beta1))
                    s.assign(s * beta2_sq + g * d_lr * (1.0 - beta2_sq))
                    self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                else:
                    momentum.assign(momentum * self.beta1 + g * (1.0 - self.beta1))
            else:
                noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                def true_fn():
                    return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                def false_fn():
                    return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                if self.DAdapt:
                    beta2_sq = math.sqrt(self.beta2)
                    pos_momentum.assign(pos_momentum * self.beta1 ** 2 + g * d_lr * (1.0 - self.beta1 ** 2))
                    s.assign(s * beta2_sq + g * d_lr * (1.0 - beta2_sq))
                    self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                else:
                    pos_momentum.assign(pos_momentum * self.beta1 ** 2 + g * (1.0 - self.beta1 ** 2))
                momentum = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
            
            def true_fn2():
                hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))
            def false_fn2():
                pass
            tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
            
            de_nom = tf.maximum(hessian_moment, self.epsilon)
            
            if self.DAdapt:
                if self.sn:
                    s = tf.reshape(s, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                flat_grad = tf.reshape(g, [-1])
                flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
            
            if not self.DAdapt:
                if self.weight_decouple:
                    p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * (1.0 if self.fixed_decay else lr)))
                elif self.weight_decay > 0.0:
                    g += p * tf.cast(self.weight_decay, p.dtype)
                    
                if self.sn:
                    numerator = tf.reshape(momentum, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    norm_grad = tf.reshape(numerator / de_nom, p.shape)
                    update = tf.clip_by_value(norm_grad, clip_value_min=-self.p, clip_value_max=self.p)
                else:
                    norm_grad = momentum / de_nom
                    update = tf.clip_by_value(norm_grad, clip_value_min=-self.p, clip_value_max=self.p)
                    
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    w_norm = tf.norm(p, ord=2)
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
                    mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                    numel = tf.cast(tf.size(mask), g.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
    
                p.assign_add(update * -lr) 
                
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
                    
        if self.DAdapt:
            def update_fn():
                beta2_sq = math.sqrt(self.beta2)
                
                d = self.d0_
                self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                
                if self.lr > 0.0:
                    d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                    d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                
                self.d0_.assign(d)
                
                for p, g in zip(trainable_variables, grads):
                    if self.weight_decouple:
                        p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * (1.0 if self.fixed_decay else d_lr)))
                    elif self.weight_decay > 0.0:
                        g += p * tf.cast(self.weight_decay, p.dtype)
                    
                    if not self.pnm:
                        momentum = self.momentum[self._get_variable_index(p)]
                    else:
                        noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                        def true_fn():
                            return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                        def false_fn():
                            return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                        pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                        momentum = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                    hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                    
                    de_nom = tf.maximum(hessian_moment, self.epsilon)
                    
                    if self.sn:
                        numerator = tf.reshape(momentum, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        norm_grad = tf.reshape(numerator / de_nom, p.shape)
                        update = tf.clip_by_value(norm_grad, clip_value_min=-self.p, clip_value_max=self.p)
                    else:
                        norm_grad = momentum / de_nom
                        update = tf.clip_by_value(norm_grad, clip_value_min=-self.p, clip_value_max=self.p)
                        
                    if self.trust_ratio:
                        # Layer-wise LR adaptation
                        w_norm = tf.norm(p, ord=2)
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
                        mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                        numel = tf.cast(tf.size(mask), g.dtype)
                        factor = numel / (tf.reduce_sum(mask) + 1)
                        mask = mask * factor
                        update = update * mask
        
                    p.assign_add(update * -1.0) 
                    
                    if self.lookahead:
                        def true_fn():
                            slow_p = self.slow_momentum[self._get_variable_index(p)]
                            slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                            p.assign(slow_p)
                        
                        def false_fn():
                            pass
                    
                        tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
            
            def no_update_fn():
                pass
            
            tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
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
                "agc": self.agc,
                "cautious": self.cautious,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "DAdapt": self.DAdapt,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class SophiaG(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-4,
        beta1=0.965,
        beta2=0.99,
        rho=0.04,
        weight_decay=1e-1,
        weight_decouple=True,
        fixed_decay=False,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sophiag",
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
        self.rho = rho
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.hessian = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.hessian.append(self.add_variable_from_reference(
                                reference_variable=var, name="hessian"
                                                    ))
    
    def update_hessian(self, params, grads):
        for p, g in zip(params, grads):
            hessian = self.hessian[self._get_variable_index(p)]
            hessian.assign(hessian * self.beta2 + g * g * (1 - self.beta2))
    
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
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'Hero does not support sparse gradients')
            
            lr = tf.cast(learning_rate, p.dtype)

            if self.weight_decouple:
                p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * tf.cast(self.weight_decay, p.dtype)
                grads[self._get_variable_index(p)] = g
            
        sophiag(trainable_variables,
              grads,
              self.exp_avg,
              self.hessian,
              beta1=self.beta1,
              beta2=self.beta2,
              rho=self.rho,
              lr=learning_rate,
              weight_decay=self.weight_decay,
              maximize=self.maximize,
              )      

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "rho": self.rho,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "maximize": self.maximize,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class SophiaG_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-4,
        beta1=0.965,
        beta2=0.99,
        rho=0.04,
        weight_decay=1e-1,
        weight_decouple=True,
        fixed_decay=False,
        maximize=False,
        orthograd=False,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=False,
        pnm=False,
        subset_size=-1,
        sn=False,
        agc=False,
        cautious=False,
        d0=1e-6,
        growth_rate=float('inf'),
        DAdapt=False,
        trust_ratio=False,
        trust_clip=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sophiag_e",
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
        self.rho = rho
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.maximize = maximize
        self.orthograd = orthograd
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.subset_size = subset_size
        self.sn = sn
        self.agc = agc
        self.cautious = cautious
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_slow = []
        self.hessian = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        if self.DAdapt:
            self.s = []
            self.sk_l1 = tf.Variable(0.0)
            self.numerator_acc = tf.Variable(0.0)
            self.numerator_weighted = tf.Variable(0.0)
            self.d0_ = tf.Variable(self.d0)
            self._track_variable(self.sk_l1)
            self._track_variable(self.numerator_acc)
            self._track_variable(self.numerator_weighted)
            self._track_variable(self.d0_)
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
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                second_moment_update = tf.Variable(second_moment_update)
                self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=second_moment_update, name="hessian"
                                                        )
            else:
                self.hessian.append(self.add_variable_from_reference(
                                    reference_variable=var, name="hessian"
                                                        ))
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                    reference_variable=var, name="s"
                                        ))
    
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
        
    def update_hessian(self, params, grads):
        for p, g in zip(params, grads):
            size = tf.size(p)
            if self.sn:
                reshaped_grad = tf.reshape(g, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                second_moment_update = tf.reduce_sum(reshaped_grad * reshaped_grad, axis=1, keepdims=True)
            else:
                second_moment_update = tf.pow(g, 2)
            hessian = self.hessian[self._get_variable_index(p)]
            hessian.assign(hessian * self.beta2 + second_moment_update * (1 - self.beta2))
    
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
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'Hero does not support sparse gradients')
            
            if self.agc:
                grads[self._get_variable_index(p)] = agc(p, g)  
            
        sophiag(trainable_variables,
              grads,
              self.exp_avg,
              self.hessian,
              self.iterations,
              self.pnm,
              self.pos_momentum,
              self.neg_momentum,
              self.momentum_slow,
              self.sn,
              self.subset_size_,
              self.cautious,
              self.lookahead,
              self.slow_momentum,
              self.lookahead_blending_alpha,
              self.lookahead_merge_time,
              self.d0,
              self.growth_rate,
              self.DAdapt,
              self.trust_ratio,
              self.trust_clip,
              self.s,
              self.sk_l1,
              self.numerator_acc,
              self.numerator_weighted,
              self.d0_,
              self.weight_decouple,
              self.fixed_decay,
              beta1=self.beta1,
              beta2=self.beta2,
              rho=self.rho,
              lr=learning_rate,
              weight_decay=self.weight_decay,
              maximize=self.maximize,
              )      

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "rho": self.rho,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "maximize": self.maximize,
                "orthograd": self.orthograd,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "agc": self.agc,
                "cautious": self.cautious,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "DAdapt": self.DAdapt,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


def sophiag(params,
          grads,
          exp_avgs,
          hessian,
          step,
          pnm,
          pos_momentums,
          neg_momentums,
          momentum_slows,
          sn,
          subset_size_,
          cautious,
          lookahead,
          slow_momentums,
          lookahead_blending_alpha,
          lookahead_merge_time,
          d0,
          growth_rate,
          DAdapt,
          trust_ratio,
          trust_clip,
          s,
          sk_l1,
          numerator_acc,
          numerator_weighted,
          d0_,
          weight_decouple,
          fixed_decay,
          *,
          beta1: float,
          beta2: float,
          rho: float,
          lr: float,
          weight_decay: float,
          maximize: bool):
    
    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         step,
         pnm,
         pos_momentums,
         neg_momentums,
         sn,
         subset_size_,
         cautious,
         lookahead,
         slow_momentums,
         lookahead_blending_alpha,
         lookahead_merge_time,
         d0,
         growth_rate,
         DAdapt,
         trust_ratio,
         trust_clip,
         s,
         sk_l1,
         numerator_acc,
         numerator_weighted,
         d0_,
         weight_decouple,
         fixed_decay,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize
         )

def _single_tensor_sophiag(params,
                         grads,
                         exp_avgs,
                         hessian,
                         step,
                         pnm,
                         pos_momentums,
                         neg_momentums,
                         momentum_slows,
                         sn,
                         subset_size_,
                         cautious,
                         lookahead,
                         slow_momentums,
                         lookahead_blending_alpha,
                         lookahead_merge_time,
                         d0,
                         growth_rate,
                         DAdapt,
                         trust_ratio,
                         trust_clip,
                         s,
                         sk_l1,
                         numerator_acc,
                         numerator_weighted,
                         d0_,
                         weight_decouple,
                         fixed_decay,
                         *,
                         beta1: float,
                         beta2: float,
                         rho: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool
                         ):

    for i, param in enumerate(params):
        lr = tf.cast(lr, param.dtype)
        if DAdapt:
            d_lr = d0 * lr
            d_lr = tf.cast(d_lr, param.dtype)
            s_ = s[i]
        
        step = tf.cast(step + 1, param.dtype)
        
        size = tf.size(param)
        
        grad = grads[i] if not maximize else -grads[i]
        if not pnm:
            exp_avg = exp_avgs[i]
        hess = hessian[i]
            
        if param.dtype.is_complex:
            grad = tf.stack([tf.math.real(grad), tf.math.imag(grad)], axis=-1)
            if not pnm:
                exp_avg = tf.stack([tf.math.real(exp_avg), tf.math.imag(exp_avg)], axis=-1)
            hess = tf.stack([tf.math.real(hess), tf.math.imag(hess)], axis=-1)
            param = tf.stack([tf.math.real(param), tf.math.imag(param)], axis=-1)

        # Perform stepweight decay
        param.assign(param * (1 - lr * weight_decay))

        # Decay the first and second moment running average coefficient
        if not pnm:
            if DAdapt:
                beta2_sq = math.sqrt(beta2)
                exp_avg.assign(exp_avg * beta1 + grad * d_lr * (1 - beta1))
                s_.assign(s_ * beta2_sq + grad * d_lr * (1.0 - beta2_sq))
                sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
            else:
                exp_avg.assign(exp_avg * beta1 + grad * (1 - beta1))
        else:
            noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
            def true_fn():
                return pos_momentums[i], neg_momentums[i]
            def false_fn():
                return neg_momentums[i], pos_momentums[i]
            pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
            if param.dtype.is_complex:
                pos_momentum = tf.stack([tf.math.real(pos_momentum), tf.math.imag(pos_momentum)], axis=-1)
                neg_momentum = tf.stack([tf.math.real(neg_momentum), tf.math.imag(neg_momentum)], axis=-1)
            if DAdapt:
                beta2_sq = math.sqrt(beta2)
                pos_momentum.assign(pos_momentum * beta1 ** 2 + grad * d_lr * (1.0 - beta1 ** 2))
                s_.assign(s_ * beta2_sq + grad * d_lr * (1.0 - beta2_sq))
                sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
            else:
                pos_momentum.assign(pos_momentum * beta1 ** 2 + grad * (1.0 - beta1 ** 2))
            exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
        
        step_size = lr 
        step_size_neg = -step_size
        
        de_nom = (rho * hess + 1e-15)
        
        if DAdapt:
            if sn:
                s = tf.reshape(s, (size // subset_size_[i], subset_size_[i]))
            flat_grad = tf.reshape(grad, [-1])
            flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
            dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
            numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
        
        if not DAdapt:
            if weight_decouple:
                param.assign(param * (1.0 - tf.cast(weight_decay, param.dtype) * (1.0 if fixed_decay else lr)))
            elif weight_decay > 0.0:
                grad += param * tf.cast(weight_decay, param.dtype)
            if sn:
                numerator = tf.reshape(exp_avg, (size // subset_size_[i], subset_size_[i]))
                norm_grad = tf.reshape(numerator / de_nom, param.shape)
                ratio = tf.minimum(norm_grad, 1)
            else:
                norm_grad = exp_avg / de_nom
                ratio = tf.minimum(norm_grad, 1)
            
            if trust_ratio:
                # Layer-wise LR adaptation
                w_norm = tf.norm(param, ord=2)
                g_norm = tf.norm(ratio, ord=2)
                trust_ratio = w_norm / g_norm
                trust_ratio = tf.where(
                    w_norm > 0,
                    tf.where(g_norm > 0, trust_ratio, 1.0),
                    1.0,
                )
                if trust_clip:
                    trust_ratio = tf.minimum(trust_ratio, 1.0)
                ratio *= trust_ratio
            
            if cautious:
                mask = tf.cast(tf.math.greater(ratio * grad, 0), grad.dtype)
                numel = tf.cast(tf.size(mask), grad.dtype)
                factor = numel / (tf.reduce_sum(mask) + 1)
                mask = mask * factor
                ratio = ratio * mask
                
            param.assign_add(step_size_neg * tf.sign(exp_avg) * ratio)
            
            if lookahead:
                def true_fn():
                    slow_p = slow_momentums[i]
                    if param.dtype.is_complex:
                        slow_p = tf.stack([tf.math.real(slow_p), tf.math.imag(slow_p)], axis=-1)
                    slow_p.assign(slow_p + lookahead_blending_alpha * (param - slow_p))
                    param.assign(slow_p)
                
                def false_fn():
                    pass
            
                tf.cond(step % lookahead_merge_time == 0, true_fn, false_fn)
    
    if DAdapt:
        def update_fn(trust_ratio = trust_ratio):
            d_lr = d0 * lr
                
            beta2_sq = math.sqrt(beta2)
            
            d = d0_
            numerator_weighted.assign(numerator_weighted * beta2_sq + numerator_acc * (1.0 - beta2_sq))  # fmt: skip
            
            if lr > 0.0:
                d_hat = numerator_weighted / (1.0 - beta2_sq) * sk_l1
                d = tf.maximum(d0_, tf.minimum(d_hat, d0_ * growth_rate))
            
            d0_.assign(d)
            
            for i, param in enumerate(params):
                d_lr = tf.cast(d_lr, param.dtype)
                grad = grads[i] if not maximize else -grads[i]
                if weight_decouple:
                    param.assign(param * (1.0 - tf.cast(weight_decay, param.dtype) * (1.0 if fixed_decay else d_lr)))
                elif weight_decay > 0.0:
                    grad += param * tf.cast(weight_decay, param.dtype)
                if not pnm:
                    exp_avg = exp_avgs[i]
                hess = hessian[i]
                 
                if param.dtype.is_complex:
                    grad = tf.stack([tf.math.real(grad), tf.math.imag(grad)], axis=-1)
                    if not pnm:
                        exp_avg = tf.stack([tf.math.real(exp_avg), tf.math.imag(exp_avg)], axis=-1)
                    else:
                        noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                        def true_fn():
                            return pos_momentums[i], neg_momentums[i]
                        def false_fn():
                            return neg_momentums[i], pos_momentums[i]
                        pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                        if param.dtype.is_complex:
                            pos_momentum = tf.stack([tf.math.real(pos_momentum), tf.math.imag(pos_momentum)], axis=-1)
                            neg_momentum = tf.stack([tf.math.real(neg_momentum), tf.math.imag(neg_momentum)], axis=-1)
                        exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                    hess = tf.stack([tf.math.real(hess), tf.math.imag(hess)], axis=-1)
                    param = tf.stack([tf.math.real(param), tf.math.imag(param)], axis=-1)
                
                if sn:
                    numerator = tf.reshape(exp_avg, (size // subset_size_[i], subset_size_[i]))
                    norm_grad = tf.reshape(numerator / de_nom, param.shape)
                    ratio = tf.minimum(norm_grad, 1)
                else:
                    norm_grad = exp_avg / de_nom
                    ratio = tf.minimum(norm_grad, 1)
                    
                if trust_ratio:
                    # Layer-wise LR adaptation
                    w_norm = tf.norm(param, ord=2)
                    g_norm = tf.norm(ratio, ord=2)
                    trust_ratio = w_norm / g_norm
                    trust_ratio = tf.where(
                        w_norm > 0,
                        tf.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if trust_clip:
                        trust_ratio = tf.minimum(trust_ratio, 1.0)
                    ratio *= trust_ratio
                
                if cautious:
                    mask = tf.cast(tf.math.greater(ratio * grad, 0), grad.dtype)
                    numel = tf.cast(tf.size(mask), grad.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    ratio = ratio * mask
                
                param.assign_add(-1.0 * tf.sign(exp_avg) * ratio)
                
                if lookahead:
                    def true_fn():
                        slow_p = slow_momentums[i]
                        if param.dtype.is_complex:
                            slow_p = tf.stack([tf.math.real(slow_p), tf.math.imag(slow_p)], axis=-1)
                        slow_p.assign(slow_p + lookahead_blending_alpha * (param - slow_p))
                        param.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % lookahead_merge_time == 0, true_fn, false_fn)
        
        def no_update_fn():
            pass
        
        tf.cond(sk_l1 == 0, no_update_fn, update_fn)