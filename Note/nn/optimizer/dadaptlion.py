""" DAdaptLion
https://arxiv.org/abs/2301.07733

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from Note.nn.optimizer.galore_projector import GaLoreProjector
import math


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


def zero_power_via_newton_schulz_5(G, steps, sn=None, subset_size=None):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = tf.cast(G, tf.bfloat16)
    if G.shape[-2] > G.shape[-1]:
        X = tf.linalg.matrix_transpose(X)

    # Ensure spectral norm is at most 1
    if sn:
        size = tf.size(X)
        reshaped_X = tf.reshape(X, (size // subset_size, subset_size))
        norm = tf.sqrt(tf.reduce_sum(tf.reduce_sum(reshaped_X ** 2, axis=1, keepdims=True), axis=0, keepdims=True))
        X = X / (norm + 1e-7)
    else:
        X = X / (tf.norm(X, axis=[-2, -1], keepdims=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = tf.matmul(X, tf.linalg.matrix_transpose(X))
        B = b * A + c * tf.matmul(A, A) # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + tf.matmul(B, X)
    
    if G.shape[-2] > G.shape[-1]:
        X = tf.linalg.matrix_transpose(X)
    return X


class DAdaptLion(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        d0=1e-6,
        weight_decouple=True,
        fixed_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptlion",
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
        self.d0 = d0
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
            
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.s = []
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0_ = tf.Variable(self.d0)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0_)
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
                                                    ))
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        beta2_sq = math.sqrt(self.beta2)
        
        d_lr = self.d0_ * learning_rate
        
        for variable, grad in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError(
                    'DAdaptLion does not support sparse gradients')
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - tf.cast(self.weight_decay, variable.dtype) * (1.0 if self.fixed_decay else d_lr)))
            elif self.weight_decay > 0.0:
                grad += variable * tf.cast(self.weight_decay, variable.dtype)
            
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            s = self.s[self._get_variable_index(variable)]
            
            d_lr = tf.cast(d_lr, variable.dtype)
            
            update = tf.math.sign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            variable.assign_add(update * -d_lr)
            
            exp_avg.assign(exp_avg * self.beta2 + grad * (1.0 - self.beta2) * d_lr)
            
            self.numerator_accumulator.assign_add(tf.cast(tf.tensordot(tf.reshape(update, [-1]), tf.reshape(s, [-1]), axes=1) * d_lr, tf.float32))
            s.assign(s * beta2_sq + update * (1.0 - beta2_sq) * d_lr)
            
            self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
        
        self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_accumulator * (1.0 - beta2_sq))
        
        def update_fn():
            d = self.d0_
            if self.lr > 0.0:
                d_hat = self.numerator_weighted / ((1.0 - beta2_sq) * self.sk_l1)
                d = tf.maximum(self.d0_, d_hat)
            
            self.d0_.assign(d)
            
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
                "d0": self.d0,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class DAdaptLion_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        d0=1e-6,
        weight_decouple=True,
        fixed_decay=False,
        orthograd=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        agc=True,
        cautious=True,
        update_proj_gap=None,
        scale=None,
        projection_type=None,
        subset_size=-1,
        sn=True,
        trust_ratio=False,
        trust_clip=False,
        muon_ortho=False,
        muon_steps=5,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptlion_e",
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
        self.d0 = d0
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.orthograd = orthograd
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type
        self.subset_size = subset_size
        self.sn = sn
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.muon_ortho = muon_ortho
        self.muon_steps = muon_steps
            
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.s = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0_ = tf.Variable(self.d0)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0_)
        self.projector = []
        self.ortho_matrix = []
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
                self.exp_avg.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg"
                                                        ))
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
                                                    ))
            if self.update_proj_gap is not None and len(var.shape) == 2:
                self.projector[self._get_variable_index(var)] = GaLoreProjector(
                    rank=None,
                    update_proj_gap=self.update_proj_gap,
                    scale=self.scale,
                    projection_type=self.projection_type,
                )
                ortho_matrix = self.projector[-1].get_orthogonal_matrix(var, None, self.projection_type)
                if self.projection_type != 'full':
                    self.ortho_matrix.append(self.add_variable_from_reference(
                                    reference_variable=ortho_matrix, name="ortho_matrix"
                                                        ))
                else:
                    self.ortho_matrix.append((self.add_variable_from_reference(
                                    reference_variable=ortho_matrix[0], name="ortho_matrix"
                                                        ), self.add_variable_from_reference(
                                    reference_variable=ortho_matrix[1], name="ortho_matrix"
                                                        )))
            else:
                self.projector.append(None)
                self.ortho_matrix.append(None)
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
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)
            
        beta2_sq = math.sqrt(self.beta2)
        
        d_lr = self.d0_ * learning_rate
        
        for variable, grad in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError(
                    'DAdaptLion_e does not support sparse gradients')
            
            step = tf.cast(self.iterations + 1, variable.dtype)
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - tf.cast(self.weight_decay, variable.dtype) * (1.0 if self.fixed_decay else d_lr)))
            elif self.weight_decay > 0.0:
                grad += variable * tf.cast(self.weight_decay, variable.dtype)
            
            if self.agc:
                grads[self._get_variable_index(variable)] = agc(variable, grad)  
                grad = grads[self._get_variable_index(variable)]
            
            if not self.pnm:
                exp_avg = self.exp_avg[self._get_variable_index(variable)]
            s = self.s[self._get_variable_index(variable)]
            
            if not self.pnm:
                exp_avg.assign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            else:
                noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                def true_fn():
                    return self.pos_momentum[self._get_variable_index(variable)], self.neg_momentum[self._get_variable_index(variable)]
                def false_fn():
                    return self.neg_momentum[self._get_variable_index(variable)], self.pos_momentum[self._get_variable_index(variable)]
                pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                pos_momentum.assign(pos_momentum * self.beta1 ** 2 + grad * (1.0 - self.beta1 ** 2))
                exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                
            if self.update_proj_gap is not None and len(variable.shape) == 2:
                grad = self.projector[self._get_variable_index(variable)].project(grad, step, exp_avg)
                exp_avg = self.projector[self._get_variable_index(variable)].project(exp_avg, step)
            
            d_lr = tf.cast(d_lr, variable.dtype)
            
            update = tf.math.sign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            
            if self.update_proj_gap is not None and len(variable.shape) == 2:
                update = self.projector[self._get_variable_index(variable)].project_back(update)
            
            if self.muon_ortho and len(variable.shape) == 2:
                update = zero_power_via_newton_schulz_5(update, num_steps=self.muon_steps, sn=self.sn, subset_size=self.subset_size_[self._get_variable_index(variable)])
                
            if self.trust_ratio:
                # Layer-wise LR adaptation
                if self.sn:
                    size = tf.size(variable)
                    reshaped_p = tf.reshape(variable, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                    reshaped_update = tf.reshape(update, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                    w_norm = tf.sqrt(tf.reduce_sum(tf.reduce_sum(reshaped_p ** 2, axis=1)))
                    g_norm = tf.sqrt(tf.reduce_sum(tf.reduce_sum(reshaped_update ** 2, axis=1)))
                else:
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
                mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                numel = tf.cast(tf.size(mask), grad.dtype)
                factor = numel / (tf.reduce_sum(mask) + 1)
                mask = mask * factor
                update = update * mask
                
            variable.assign_add(update * -d_lr)
            
            if self.lookahead:
                def true_fn():
                    slow_p = self.slow_momentum[self._get_variable_index(variable)]
                    slow_p.assign(slow_p + self.lookahead_blending_alpha * (variable - slow_p))
                    variable.assign(slow_p)
                
                def false_fn():
                    pass
            
                tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
            
            exp_avg.assign(exp_avg * self.beta2 + grad * (1.0 - self.beta2) * d_lr)
            
            self.numerator_accumulator.assign_add(tf.cast(tf.tensordot(tf.reshape(update, [-1]), tf.reshape(s, [-1]), axes=1) * d_lr, tf.float32))
            s.assign(s * beta2_sq + update * (1.0 - beta2_sq) * d_lr)
            
            self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
        
        self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_accumulator * (1.0 - beta2_sq))
        
        def update_fn():
            d = self.d0_
            if self.lr > 0.0:
                d_hat = self.numerator_weighted / ((1.0 - beta2_sq) * self.sk_l1)
                d = tf.maximum(self.d0_, d_hat)
            
            self.d0_.assign(d)
            
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
                "d0": self.d0,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "orthograd": self.orthograd,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "cautious": self.cautious,
                "update_proj_gap": self.update_proj_gap,
                "scale": self.scale,
                "projection_type": self.projection_type,
                "projector": self.projector,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
                "muon_ortho": self.muon_ortho,
                "muon_steps": self.muon_steps,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
