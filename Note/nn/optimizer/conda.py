""" Conda
https://arxiv.org/abs/2509.24218

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from Note.nn.optimizer.galore_projector import GaLoreProjector
import math


class Conda(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        rank=None,
        update_proj_gap=None,
        scale=None,
        projection_type=None,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="conda",
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
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.projector = []
        self.ortho_matrix = []
        for var in var_list:
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
                self.projector[-1].ortho_matrix = self.ortho_matrix[-1]
            else:
                self.projector.append(None)
                self.ortho_matrix.append(None)
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.projector.append(None)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
                
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
        
        step_size = lr * bias_correction2_sq / bias_correction1
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Conda does not support sparse gradients')
        
        if self.maximize:
            gradient = -gradient
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        
        if self.update_proj_gap is not None and len(variable.shape) == 2:
            gradient = self.projector[self._get_variable_index(variable)].project(gradient, step, exp_avg)
            exp_avg = self.projector[self._get_variable_index(variable)].project(exp_avg, step)
        
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + gradient * gradient * (1.0 - self.beta2))
        
        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        norm_grad = exp_avg / de_nom
        
        if self.update_proj_gap is not None and len(variable.shape) == 2:
            norm_grad = self.projector[self._get_variable_index(variable)].project_back(norm_grad)
        
        variable.assign_add(norm_grad * -step_size)
        
        variable.assign(variable * (1.0 - tf.cast(self.weight_decay, variable.dtype) * lr))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "rank": self.rank,
                "update_proj_gap": self.update_proj_gap,
                "scale": self.scale,
                "projection_type": self.projection_type,
                "projector": self.projector,
                "maximize": self.maximize,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


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
    
    
class Conda_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        beta3=0.9999,
        epsilon=1e-8,
        weight_decay=0.0,
        rank=None,
        update_proj_gap=None,
        scale=None,
        projection_type=None,
        maximize=False,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        subset_size=-1,
        sn=True,
        agc=True,
        cautious=True,
        aem=True,
        alpha=5.0,
        t_alpha_beta3=None,
        d0=1e-6,
        growth_rate=float('inf'),
        DAdapt=True,
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
        name="conda_e",
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
        self.beta3 = beta3
        self.epsilon = epsilon
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type
        self.maximize = maximize
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.subset_size = subset_size
        self.sn = sn
        self.agc = agc
        self.cautious = cautious
        self.aem = aem
        self.alpha = alpha
        self.t_alpha_beta3 = t_alpha_beta3
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
        self.exp_avg_sq = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        self.projector = []
        self.ortho_matrix = []
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
                self.projector[-1].ortho_matrix = self.ortho_matrix[-1]
            else:
                self.projector.append(None)
                self.ortho_matrix.append(None)
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
                self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=second_moment_update, name="exp_avg_sq"
                                                        )
            else:
                self.exp_avg_sq.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        ))
            if self.aem:
                self.exp_avg_slow.append(self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_slow"
                                        ))
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=var, name="s"
                                                            ))
            self.projector.append(None)
            
    @staticmethod
    def schedule_alpha(t_alpha_beta3, step, alpha):
        return alpha if t_alpha_beta3 is None else tf.minimum(step * alpha / t_alpha_beta3, alpha)
    
    @staticmethod
    def schedule_beta3(t_alpha_beta3, step, beta1, beta3):
        if t_alpha_beta3 is None:
            return beta3
    
        log_beta1, log_beta3 = tf.math.log(beta1), tf.math.log(beta3)
    
        return tf.minimum(
            tf.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )
    
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
                    'Conda does not support sparse gradients')
                
            lr = tf.cast(learning_rate, p.dtype)
                    
            step = tf.cast(self.iterations + 1, p.dtype)
            
            size = tf.size(p)
                
            if self.aem:
                beta1 = tf.cast(self.beta1, p.dtype)
                beta3 = tf.cast(self.beta3, p.dtype)
                
                alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
                
                clip = tf.pow(step, 0.25)
            
            if self.agc:
                grads[self._get_variable_index(p)] = agc(p, g) 
                g = grads[self._get_variable_index(p)]
            
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
            
            if self.DAdapt:
                d_lr = self.d0 * lr * bias_correction2_sq / bias_correction1
            
            step_size = lr * bias_correction2_sq / bias_correction1
        
            if self.maximize:
                g = -g
        
            if not self.pnm:
                exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            if self.aem:
                exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
            
            if self.DAdapt:
                de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
                
                s = self.s[self._get_variable_index(p)]
                if self.sn:
                    s = tf.reshape(s, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
            
                flat_grad = tf.reshape(g, [-1])
                flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
                
                d_lr = tf.cast(d_lr, dtype=p.dtype)
                
            if not self.aem:
                normed_grad = g
            else:
                normed_grad = tf.clip_by_value(
                    g / tf.maximum(tf.sqrt(exp_avg_sq), self.epsilon if self.epsilon is not None else 1e-8),
                    clip_value_min=-clip,
                    clip_value_max= clip,
                )
                
            if not self.pnm:
                if self.DAdapt:
                    beta2_sq = math.sqrt(self.beta2)
                    exp_avg.assign(exp_avg * self.beta1 + normed_grad * d_lr * (1.0 - self.beta1))
                    s.assign(s * beta2_sq + g * d_lr * (1.0 - beta2_sq))
                    self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                else:
                    exp_avg.assign(exp_avg * self.beta1 + normed_grad * (1.0 - self.beta1))
            else:
                noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                def true_fn():
                    return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                def false_fn():
                    return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                if self.DAdapt:
                    beta2_sq = math.sqrt(self.beta2)
                    pos_momentum.assign(pos_momentum * self.beta1 ** 2 + normed_grad * d_lr * (1.0 - self.beta1 ** 2))
                    s.assign(s * beta2_sq + g * d_lr * (1.0 - beta2_sq))
                    self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                else:
                    pos_momentum.assign(pos_momentum * self.beta1 ** 2 + normed_grad * (1.0 - self.beta1 ** 2))
                exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                
            if self.aem:
                exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                
            if not self.DAdapt:
                if self.aem:
                    exp_avg += exp_avg_slow * alpha_t
                    
                if self.update_proj_gap is not None and len(p.shape) == 2:
                    g = self.projector[self._get_variable_index(p)].project(g, step, exp_avg)
                    exp_avg = self.projector[self._get_variable_index(p)].project(exp_avg, step)
                    
                if self.sn:
                    reshaped_grad = tf.reshape(g, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                else:
                    second_moment_update = tf.pow(g, 2)
        
                exp_avg_sq.assign(exp_avg_sq * self.beta2 + second_moment_update * (1.0 - self.beta2))
        
                de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
                if self.sn:
                    numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    norm_grad = tf.reshape(numerator / de_nom, p.shape)
                else:
                    norm_grad = exp_avg / de_nom
        
                if self.update_proj_gap is not None and len(p.shape) == 2:
                    norm_grad = self.projector[self._get_variable_index(p)].project_back(norm_grad)
                    
                update = norm_grad
                    
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                    numel = tf.cast(tf.size(mask), g.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
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
        
                p.assign_add(update * -step_size)
                
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
        
                p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * lr))
                
        def update_fn():
            lr = learning_rate
            step = self.iterations + 1
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
            d_lr = self.d0 * lr * bias_correction2_sq / bias_correction1
            
            beta2_sq = math.sqrt(self.beta2)
            
            d = self.d0_
            self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
            
            if self.lr > 0.0:
                d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
            
            self.d0_.assign(d)
            
            for p, g in zip(trainable_variables, grads):
                if not self.pnm:
                    exp_avg = self.exp_avg[self._get_variable_index(p)]
                exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                if self.aem:
                    exp_avg_slow = self.momentum_slow[self._get_variable_index(p)]
                    alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                
                if self.pnm:
                    noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                
                if self.aem:
                    exp_avg += exp_avg_slow * alpha_t
                
                step_size = tf.cast(d_lr, p.dtype)
    
                if self.update_proj_gap is not None and len(p.shape) == 2:
                    g = self.projector[self._get_variable_index(p)].project(g, step, exp_avg)
                    exp_avg = self.projector[self._get_variable_index(p)].project(exp_avg, step)
                    
                de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
                
                if self.sn:
                    numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    norm_grad = tf.reshape(numerator / de_nom, p.shape)
                else:
                    norm_grad = exp_avg / de_nom
    
                if self.update_proj_gap is not None and len(p.shape) == 2:
                    norm_grad = self.projector[self._get_variable_index(p)].project_back(norm_grad)
                    
                update = norm_grad
                
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                    numel = tf.cast(tf.size(mask), g.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
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
    
                p.assign_add(update * -step_size)
                
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
                
                p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * lr))
        
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
                "beta3": self.beta3,
                "epsilon": self.epsilon,
                "rank": self.rank,
                "update_proj_gap": self.update_proj_gap,
                "scale": self.scale,
                "projection_type": self.projection_type,
                "maximize": self.maximize,
                "projector": self.projector,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "agc": self.agc,
                "cautious": self.cautious,
                "aem": self.aem,
                "alpha": self.alpha,
                "t_alpha_beta3": self.t_alpha_beta3,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "DAdapt": self.DAdapt,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass