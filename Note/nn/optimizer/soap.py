""" SOAP
https://arxiv.org/abs/2409.11321

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from itertools import chain
import math


def merge_small_dims(shape_to_merge, max_dim):
    r"""Merge small dimensions.

        If there are some small dimensions, we collapse them
            e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
            [1, 2, 768, 1, 2048] --> [2, 768, 2048].

    :param shape_to_merge: Union[List[int], torch.Size]. Shape to merge small dimensions.
    :param max_dim: int. Maximal dimension of output shape used in merging.
    """
    merged_shape = []

    product = 1
    for dim in shape_to_merge:
        product *= dim
        if product > max_dim:
            merged_shape.append(product // dim)
            product = dim

    merged_shape.append(product)

    return merged_shape if len(merged_shape) > 1 else [1]


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


class SOAP(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=3e-3,
        beta1=0.95,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=1e-2,
        shampoo_beta=None,
        precondition_frequency=10,
        max_precondition_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        correct_bias=True,
        normalize_gradient=False,
        data_format='channels_last',
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="soap",
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
        self.shampoo_beta = shampoo_beta
        self.precondition_frequency = precondition_frequency
        self.max_precondition_dim = max_precondition_dim
        self.merge_dims = merge_dims
        self.precondition_1d = precondition_1d
        self.correct_bias = correct_bias
        self.normalize_gradient = normalize_gradient
        self.data_format = data_format
    
    def reset(self):
        self.GG = {}
        self.Q = {}
        self.matrices = {}
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.init_pre_conditioner(var, self.precondition_frequency, self.shampoo_beta if self.shampoo_beta is not None else self.beta2,
                                      self.max_precondition_dim, self.precondition_1d, self.merge_dims)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.GG = {}
        self.Q = {}
        self.matrices = {}
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.init_pre_conditioner(var, self.precondition_frequency, self.shampoo_beta if self.shampoo_beta is not None else self.beta2,
                                      self.max_precondition_dim, self.precondition_1d, self.merge_dims)
    
    def project(
        self,
        param,
        grad,
        merge_dims = False,
        max_precondition_dim = 10000,
        project_type = 'forward',
    ):
        original_shape = grad.shape

        if merge_dims:
            if self.data_format == 'channels_first' and len(grad.shape) == 4:
                permuted_shape = tf.transpose(grad, (0, 2, 3, 1)).shape 

            grad = tf.reshape(grad, merge_small_dims(grad.shape, max_precondition_dim))

        for mat in self.Q[self._get_variable_index(param)]:
            if mat.shape != []:
                grad = tf.tensordot(grad, mat, axes=[[0], [0 if project_type == 'forward' else 1]])
            else:
                grad = tf.transpose(grad, [*list(range(1, len(grad.shape))), 0])

        if merge_dims:
            if self.data_format == 'channels_first' and len(original_shape) == 4:
                grad = tf.transpose(tf.reshape(grad, permuted_shape), (0, 3, 1, 2))
            else:
                grad = tf.reshape(grad, original_shape)

        return grad
        
    def get_orthogonal_matrix(self, param, mat):
        for i, m in enumerate(mat):
            if m.shape==[]:
                continue
            else:
                try:
                    _, q = tf.linalg.eigh(m + 1e-30 * tf.eye(m.shape[0], dtype=m.dtype))
                except Exception:  # pragma: no cover
                    _, q = tf.linalg.eigh(
                        tf.cast(m, tf.float64) + 1e-30 * tf.eye(m.shape[0], dtype=tf.float64)
                    )
                    q = tf.cast(q, m.dtype)

                q = tf.reverse(q, axis=[1])
                
                self.matrices[self._get_variable_index(param)][i].assign(q)
    
    def get_orthogonal_matrix_qr(self, param, max_precondition_dim = 10000, merge_dims = False):
        r"""Compute the eigen-bases of the pre-conditioner using one round of power iteration."""
        orig_shape = self.exp_avg_sq[self._get_variable_index(param)].shape
        if self.data_format == 'channels_first' and len(orig_shape) == 4:
            permuted_shape = tf.transpose(self.exp_avg_sq[self._get_variable_index(param)], (0, 2, 3, 1)).shape

        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(param)]
        if merge_dims:
            exp_avg_sq = tf.reshape(exp_avg_sq, merge_small_dims(exp_avg_sq.shape, max_precondition_dim))

        for ind, (m, o) in enumerate(zip(self.GG[self._get_variable_index(param)], self.Q[self._get_variable_index(param)])):
            if m.shape==[]:
                continue
            else:
                est_eig = tf.linalg.diag_part(tf.matmul(tf.transpose(o), tf.matmul(m, o)))
                sort_idx = tf.argsort(est_eig, direction='DESCENDING')
                exp_avg_sq = tf.gather(exp_avg_sq, sort_idx, axis=ind)
    
                power_iter = tf.matmul(m, tf.gather(o, sort_idx, axis=1))
    
                # Compute QR decomposition
                # We cast to float32 because:
                #  - torch.linalg.qr does not have support for types like bfloat16 as of PyTorch 2.5.1
                #  - the correctness / numerical stability of the Q orthogonality is important for the stability
                #    of the optimizer
                q, _ = tf.linalg.qr(tf.cast(power_iter, tf.float32))
                q = tf.cast(q, power_iter.dtype)
    
                self.matrices[self._get_variable_index(param)][ind].assign(q)

        if merge_dims:
            if self.data_format == 'channels_first' and len(orig_shape) == 4:
                exp_avg_sq.assign(tf.transpose(tf.reshape(exp_avg_sq, permuted_shape), (0, 3, 1, 2)))
            else:
                exp_avg_sq.assign(tf.reshape(exp_avg_sq, orig_shape))
    
    def init_pre_conditioner(
        self,
        var,
        precondition_frequency = 10,
        shampoo_beta = 0.95,
        max_precondition_dim = 10000,
        precondition_1d = False,
        merge_dims = False,
    ):
        param_idx = self._get_variable_index(var)
        self.GG[param_idx] = []
        self.Q[param_idx] = []
        self.matrices[param_idx] = []
        if len(var.shape) == 1:
            if not precondition_1d or var.shape[0] > max_precondition_dim:
                self.GG[param_idx].append(tf.Variable(tf.ones(())))
                self.Q[param_idx].append(tf.Variable(tf.ones(())))
                self.matrices[param_idx].append(tf.Variable(tf.ones(())))
                self._track_variable(self.GG[param_idx][-1])
                self._track_variable(self.Q[param_idx][-1])
                self._track_variable(self.matrices[param_idx][-1])
            else:
                self.GG[param_idx].append(tf.Variable(tf.zeros((var.shape[0], var.shape[0]), dtype=var.dtype)))
                self.Q[param_idx].append(tf.Variable(tf.zeros((var.shape[0], var.shape[0]), dtype=var.dtype)))
                self.matrices[param_idx].append(tf.Variable(tf.zeros((var.shape[0], var.shape[0]), dtype=var.dtype)))
                self._track_variable(self.GG[param_idx][-1])
                self._track_variable(self.Q[param_idx][-1])
                self._track_variable(self.matrices[param_idx][-1])
        else:
            if merge_dims:
                var = tf.reshape(var, merge_small_dims(var.shape, max_precondition_dim))

            for sh in var.shape:
                if sh > max_precondition_dim:
                    self.GG[param_idx].append(tf.Variable(tf.ones(())))
                    self.Q[param_idx].append(tf.Variable(tf.ones(())))
                    self.matrices[param_idx].append(tf.Variable(tf.ones(())))
                    self._track_variable(self.GG[param_idx][-1])
                    self._track_variable(self.Q[param_idx][-1])
                    self._track_variable(self.matrices[param_idx][-1])
                else:
                    self.GG[param_idx].append(tf.Variable(tf.zeros((sh, sh), dtype=var.dtype)))
                    self.Q[param_idx].append(tf.Variable(tf.zeros((sh, sh), dtype=var.dtype)))
                    self.matrices[param_idx].append(tf.Variable(tf.zeros((sh, sh), dtype=var.dtype)))
                    self._track_variable(self.GG[param_idx][-1])
                    self._track_variable(self.Q[param_idx][-1])
                    self._track_variable(self.matrices[param_idx][-1])
    
    def update_pre_conditioner(
        self,
        param,
        grad,
        step,
        max_precondition_dim = 10000,
        precondition_1d = False,
        merge_dims = False,
    ):
        beta = self.shampoo_beta if self.shampoo_beta is not None else self.beta2
        
        if len(grad.shape) == 1:
            if precondition_1d and grad.shape[0] <= max_precondition_dim:
                outer_product = tf.tensordot(
                tf.expand_dims(grad, 1),
                tf.expand_dims(grad, 0),
                axes=1
                )
                GG = self.GG[self._get_variable_index(param)][0]
                GG.assign(GG * beta + tf.cast(outer_product, GG.dtype) * (1.0 - beta))
        else:
            if merge_dims:
                grad = tf.reshape(grad, merge_small_dims(grad.shape, max_precondition_dim))

            for idx, dim in enumerate(grad.shape):
                if dim <= max_precondition_dim:
                    outer_product = tf.tensordot(
                        grad,
                        grad,
                        axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )

                    GG = self.GG[self._get_variable_index(param)][idx]
                    GG.assign(
                        GG * beta + tf.cast(outer_product, GG.dtype) * (1.0 - beta)
                    )
        
        def true_fn1():
            self.get_orthogonal_matrix(param, self.GG[self._get_variable_index(param)])
            if len(self.Q[self._get_variable_index(param)]) == 1:
                self.Q[self._get_variable_index(param)][0].assign(self.matrices[self._get_variable_index(param)][0])
            else:
                for i in range(len(self.Q[self._get_variable_index(param)])):
                    self.Q[self._get_variable_index(param)][i].assign(self.matrices[self._get_variable_index(param)][i])
        
        def false_fn1():
            pass
        
        tf.cond(self.iterations == 0, true_fn1, false_fn1)
        
        def true_fn2():
            self.get_orthogonal_matrix_qr(param, max_precondition_dim, merge_dims)
            if len(self.Q[self._get_variable_index(param)]) == 1:
                self.Q[self._get_variable_index(param)][0].assign(self.matrices[self._get_variable_index(param)][0])
            else:
                for i in range(len(self.Q[self._get_variable_index(param)])):
                    self.Q[self._get_variable_index(param)][i].assign(self.matrices[self._get_variable_index(param)][i])
        
        def false_fn2():
            pass
        
        tf.cond(tf.logical_and(self.iterations + 1 > 0, (self.iterations + 1) % self.precondition_frequency == 0), true_fn2, false_fn2)
    
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
                    'SOAP does not support sparse gradients')
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            lr = tf.cast(learning_rate, p.dtype)
            
            def true_fn():
                self.update_pre_conditioner(
                    p,
                    g,
                    step=step,
                    max_precondition_dim=self.max_precondition_dim,
                    precondition_1d=self.precondition_1d,
                    merge_dims=self.merge_dims,
                )
            
            def false_fn():
                pass
            
            tf.cond(self.iterations == 0, true_fn, false_fn)

            grad_projected = self.project(
                p, g, merge_dims=self.merge_dims, max_precondition_dim=self.max_precondition_dim
            )

            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]

            exp_avg.assign(exp_avg * self.beta1 + g * (1.0 - self.beta1))
            exp_avg_sq.assign(exp_avg_sq * self.beta2 + tf.square(grad_projected) * (1.0 - self.beta2))

            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon

            exp_avg_projected = self.project(
                p, exp_avg, merge_dims=self.merge_dims, max_precondition_dim=self.max_precondition_dim
            )

            step_size = lr
            if self.correct_bias:
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)

                step_size *= bias_correction2_sq / bias_correction1

            norm_grad = self.project(
                p,
                exp_avg_projected / de_nom,
                merge_dims=self.merge_dims,
                max_precondition_dim=self.max_precondition_dim,
                project_type='backward',
            )

            if self.normalize_gradient:
                norm_grad = norm_grad / (tf.sqrt((tf.reduce_mean(tf.square(norm_grad)))) + self.epsilon)

            p.assign_add(norm_grad * -step_size)
            
            if self.weight_decay > 0.0:
                p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * lr))

            self.update_pre_conditioner(
                p,
                g,
                step=step,
                max_precondition_dim=self.max_precondition_dim,
                merge_dims=self.merge_dims,
                precondition_1d=self.precondition_1d,
            )    

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "shampoo_beta": self.shampoo_beta,
                "precondition_frequency": self.precondition_frequency,
                "max_precondition_dim": self.max_precondition_dim,
                "merge_dims": self.merge_dims,
                "precondition_1d": self.precondition_1d,
                "correct_bias": self.correct_bias,
                "normalize_gradient": self.normalize_gradient,
                "data_format": self.data_format,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class SOAP_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=3e-3,
        beta1=0.95,
        beta2=0.95,
        beta3=0.9999,
        epsilon=1e-8,
        weight_decay=1e-2,
        shampoo_beta=None,
        precondition_frequency=10,
        max_precondition_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        correct_bias=True,
        normalize_gradient=False,
        data_format='channels_last',
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
        name="soap_e",
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
        self.shampoo_beta = shampoo_beta
        self.precondition_frequency = precondition_frequency
        self.max_precondition_dim = max_precondition_dim
        self.merge_dims = merge_dims
        self.precondition_1d = precondition_1d
        self.correct_bias = correct_bias
        self.normalize_gradient = normalize_gradient
        self.data_format = data_format
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
        self.GG = {}
        self.Q = {}
        self.matrices = {}
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
                self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=second_moment_update, name="exp_avg_sq"
                                                        )
            else:
                self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="exp_avg_sq"
                                                        )
            self.init_pre_conditioner(var, self.precondition_frequency, self.shampoo_beta if self.shampoo_beta is not None else self.beta2,
                                      self.max_precondition_dim, self.precondition_1d, self.merge_dims)
            if self.aem:
                self.exp_avg_slow.append(self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_slow"
                                        ))
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=var, name="s"
                                                            ))
    
    def project(
        self,
        param,
        grad,
        merge_dims = False,
        max_precondition_dim = 10000,
        project_type = 'forward',
    ):
        original_shape = grad.shape

        if merge_dims:
            if self.data_format == 'channels_first' and len(grad.shape) == 4:
                permuted_shape = tf.transpose(grad, (0, 2, 3, 1)).shape 

            grad = tf.reshape(grad, merge_small_dims(grad.shape, max_precondition_dim))

        for mat in self.Q[self._get_variable_index(param)]:
            if mat.shape != []:
                grad = tf.tensordot(grad, mat, axes=[[0], [0 if project_type == 'forward' else 1]])
            else:
                grad = tf.transpose(grad, [*list(range(1, len(grad.shape))), 0])

        if merge_dims:
            if self.data_format == 'channels_first' and len(original_shape) == 4:
                grad = tf.transpose(tf.reshape(grad, permuted_shape), (0, 3, 1, 2))
            else:
                grad = tf.reshape(grad, original_shape)

        return grad
    
    def get_orthogonal_matrix(self, param, mat):
        for i, m in enumerate(mat):
            if m.shape==[]:
                continue
            else:
                try:
                    _, q = tf.linalg.eigh(m + 1e-30 * tf.eye(m.shape[0], dtype=m.dtype))
                except Exception:  # pragma: no cover
                    _, q = tf.linalg.eigh(
                        tf.cast(m, tf.float64) + 1e-30 * tf.eye(m.shape[0], dtype=tf.float64)
                    )
                    q = tf.cast(q, m.dtype)

                q = tf.reverse(q, axis=[1])
                
                self.matrices[self._get_variable_index(param)][i].assign(q)
    
    def get_orthogonal_matrix_qr(self, param, max_precondition_dim = 10000, merge_dims = False):
        r"""Compute the eigen-bases of the pre-conditioner using one round of power iteration."""
        orig_shape = self.exp_avg_sq[self._get_variable_index(param)].shape
        if self.data_format == 'channels_first' and len(orig_shape) == 4:
            permuted_shape = tf.transpose(self.exp_avg_sq[self._get_variable_index(param)], (0, 2, 3, 1)).shape

        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(param)]
        if merge_dims:
            exp_avg_sq = tf.reshape(exp_avg_sq, merge_small_dims(exp_avg_sq.shape, max_precondition_dim))

        for ind, (m, o) in enumerate(zip(self.GG[self._get_variable_index(param)], self.Q[self._get_variable_index(param)])):
            if m.shape==[]:
                continue
            else:
                est_eig = tf.linalg.diag_part(tf.matmul(tf.transpose(o), tf.matmul(m, o)))
                sort_idx = tf.argsort(est_eig, direction='DESCENDING')
                exp_avg_sq = tf.gather(exp_avg_sq, sort_idx, axis=ind)
    
                power_iter = tf.matmul(m, tf.gather(o, sort_idx, axis=1))
    
                # Compute QR decomposition
                # We cast to float32 because:
                #  - torch.linalg.qr does not have support for types like bfloat16 as of PyTorch 2.5.1
                #  - the correctness / numerical stability of the Q orthogonality is important for the stability
                #    of the optimizer
                q, _ = tf.linalg.qr(tf.cast(power_iter, tf.float32))
                q = tf.cast(q, power_iter.dtype)
    
                self.matrices[self._get_variable_index(param)][ind].assign(q)

        if merge_dims:
            if self.data_format == 'channels_first' and len(orig_shape) == 4:
                exp_avg_sq.assign(tf.transpose(tf.reshape(exp_avg_sq, permuted_shape), (0, 3, 1, 2)))
            else:
                exp_avg_sq.assign(tf.reshape(exp_avg_sq, orig_shape))
    
    def init_pre_conditioner(
        self,
        var,
        precondition_frequency = 10,
        shampoo_beta = 0.95,
        max_precondition_dim = 10000,
        precondition_1d = False,
        merge_dims = False,
    ):
        param_idx = self._get_variable_index(var)
        self.GG[param_idx] = []
        self.Q[param_idx] = []
        self.matrices[param_idx] = []
        if len(var.shape) == 1:
            if not precondition_1d or var.shape[0] > max_precondition_dim:
                self.GG[param_idx].append(tf.Variable(tf.ones(())))
                self.Q[param_idx].append(tf.Variable(tf.ones(())))
                self.matrices[param_idx].append(tf.Variable(tf.ones(())))
                self._track_variable(self.GG[param_idx][-1])
                self._track_variable(self.Q[param_idx][-1])
                self._track_variable(self.matrices[param_idx][-1])
            else:
                self.GG[param_idx].append(tf.Variable(tf.zeros((var.shape[0], var.shape[0]), dtype=var.dtype)))
                self.Q[param_idx].append(tf.Variable(tf.zeros((var.shape[0], var.shape[0]), dtype=var.dtype)))
                self.matrices[param_idx].append(tf.Variable(tf.zeros((var.shape[0], var.shape[0]), dtype=var.dtype)))
                self._track_variable(self.GG[param_idx][-1])
                self._track_variable(self.Q[param_idx][-1])
                self._track_variable(self.matrices[param_idx][-1])
        else:
            if merge_dims:
                var = tf.reshape(var, merge_small_dims(var.shape, max_precondition_dim))

            for sh in var.shape:
                if sh > max_precondition_dim:
                    self.GG[param_idx].append(tf.Variable(tf.ones(())))
                    self.Q[param_idx].append(tf.Variable(tf.ones(())))
                    self.matrices[param_idx].append(tf.Variable(tf.ones(())))
                    self._track_variable(self.GG[param_idx][-1])
                    self._track_variable(self.Q[param_idx][-1])
                    self._track_variable(self.matrices[param_idx][-1])
                else:
                    self.GG[param_idx].append(tf.Variable(tf.zeros((sh, sh), dtype=var.dtype)))
                    self.Q[param_idx].append(tf.Variable(tf.zeros((sh, sh), dtype=var.dtype)))
                    self.matrices[param_idx].append(tf.Variable(tf.zeros((sh, sh), dtype=var.dtype)))
                    self._track_variable(self.GG[param_idx][-1])
                    self._track_variable(self.Q[param_idx][-1])
                    self._track_variable(self.matrices[param_idx][-1])
    
    def update_pre_conditioner(
        self,
        param,
        grad,
        step,
        max_precondition_dim = 10000,
        precondition_1d = False,
        merge_dims = False,
    ):
        beta = self.shampoo_beta if self.shampoo_beta is not None else self.beta2
        
        if len(grad.shape) == 1:
            if precondition_1d and grad.shape[0] <= max_precondition_dim:
                outer_product = tf.tensordot(
                tf.expand_dims(grad, 1),
                tf.expand_dims(grad, 0),
                axes=1
                )
                GG = self.GG[self._get_variable_index(param)][0]
                GG.assign(GG * beta + tf.cast(outer_product, GG.dtype) * (1.0 - beta))
        else:
            if merge_dims:
                grad = tf.reshape(grad, merge_small_dims(grad.shape, max_precondition_dim))

            for idx, dim in enumerate(grad.shape):
                if dim <= max_precondition_dim:
                    outer_product = tf.tensordot(
                        grad,
                        grad,
                        axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )

                    GG = self.GG[self._get_variable_index(param)][idx]
                    GG.assign(
                        GG * beta + tf.cast(outer_product, GG.dtype) * (1.0 - beta)
                    )
        
        def true_fn1():
            self.get_orthogonal_matrix(param, self.GG[self._get_variable_index(param)])
            if len(self.Q[self._get_variable_index(param)]) == 1:
                self.Q[self._get_variable_index(param)][0].assign(self.matrices[self._get_variable_index(param)][0])
            else:
                for i in range(len(self.Q[self._get_variable_index(param)])):
                    self.Q[self._get_variable_index(param)][i].assign(self.matrices[self._get_variable_index(param)][i])
        
        def false_fn1():
            pass
        
        tf.cond(self.iterations == 0, true_fn1, false_fn1)
        
        def true_fn2():
            self.get_orthogonal_matrix_qr(param, max_precondition_dim, merge_dims)
            if len(self.Q[self._get_variable_index(param)]) == 1:
                self.Q[self._get_variable_index(param)][0].assign(self.matrices[self._get_variable_index(param)][0])
            else:
                for i in range(len(self.Q[self._get_variable_index(param)])):
                    self.Q[self._get_variable_index(param)][i].assign(self.matrices[self._get_variable_index(param)][i])
        
        def false_fn2():
            pass
        
        tf.cond(tf.logical_and(self.iterations + 1 > 0, (self.iterations + 1) % self.precondition_frequency == 0), true_fn2, false_fn2)
    
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
                    'SOAP_e does not support sparse gradients')
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            lr = tf.cast(learning_rate, p.dtype)
            
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

            def true_fn():
                self.update_pre_conditioner(
                    p,
                    g,
                    step=step,
                    max_precondition_dim=self.max_precondition_dim,
                    precondition_1d=self.precondition_1d,
                    merge_dims=self.merge_dims,
                )
            
            def false_fn():
                pass
            
            tf.cond(self.iterations == 0, true_fn, false_fn)

            grad_projected = self.project(
                p, g, merge_dims=self.merge_dims, max_precondition_dim=self.max_precondition_dim
            )
            
            if not self.pnm:
                exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            if self.aem:
                exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
            
            step_size = lr
            if self.correct_bias:
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)

                step_size *= bias_correction2_sq / bias_correction1
                
                if self.DAdapt:
                    d_lr = self.d0 * lr * bias_correction2_sq / bias_correction1
            else:
                if self.DAdapt:
                    d_lr = self.d0 * lr
            
            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            if self.DAdapt:
                s = self.s[self._get_variable_index(p)]
            
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
                    
                if self.sn:
                    reshaped_grad = tf.reshape(grad_projected, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                else:
                    second_moment_update = tf.pow(grad_projected, 2)
    
                exp_avg_sq.assign(exp_avg_sq * self.beta2 + second_moment_update * (1.0 - self.beta2))
    
                exp_avg_projected = self.project(
                    p, exp_avg, merge_dims=self.merge_dims, max_precondition_dim=self.max_precondition_dim
                )
                
                if self.sn:
                    numerator = tf.reshape(exp_avg_projected, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    norm_grad = tf.reshape(numerator / de_nom, p.shape)
                else:
                    norm_grad = exp_avg_projected / de_nom
    
                norm_grad = self.project(
                    p,
                    norm_grad,
                    merge_dims=self.merge_dims,
                    max_precondition_dim=self.max_precondition_dim,
                    project_type='backward',
                )
    
                if self.normalize_gradient:
                    update = tf.sqrt(norm_grad / tf.reduce_mean(tf.square(norm_grad))) + self.epsilon
                else:
                    update = norm_grad
                
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                    numel = tf.cast(tf.size(mask), g.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    if self.sn:
                        w_norm = tf.reshape(p, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        g_norm = tf.reshape(update, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
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
                
                if self.weight_decay > 0.0:
                    p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * lr))
    
                self.update_pre_conditioner(
                    p,
                    g,
                    step=step,
                    max_precondition_dim=self.max_precondition_dim,
                    merge_dims=self.merge_dims,
                    precondition_1d=self.precondition_1d,
                )
        
        def update_fn():
            lr = learning_rate
            if self.correct_bias:
                step = self.iterations + 1
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
                d_lr = self.d0 * lr * bias_correction2_sq / bias_correction1
            else:
                d_lr = self.d0 * lr
            
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
                
                de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
    
                exp_avg_projected = self.project(
                    p, exp_avg, merge_dims=self.merge_dims, max_precondition_dim=self.max_precondition_dim
                )
                
                if self.sn:
                    numerator = tf.reshape(exp_avg_projected, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    norm_grad = tf.reshape(numerator / de_nom, p.shape)
                else:
                    norm_grad = exp_avg_projected / de_nom
    
                norm_grad = self.project(
                    p,
                    norm_grad,
                    merge_dims=self.merge_dims,
                    max_precondition_dim=self.max_precondition_dim,
                    project_type='backward',
                )
    
                if self.normalize_gradient:
                    update = norm_grad / (tf.sqrt((tf.reduce_mean(tf.square(norm_grad)))) + self.epsilon)
                else:
                    update = norm_grad
                
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                    numel = tf.cast(tf.size(mask), g.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    if self.sn:
                        w_norm = tf.reshape(p, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        g_norm = tf.reshape(update, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
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
                
                if self.weight_decay > 0.0:
                    p.assign(p * (1.0 - tf.cast(self.weight_decay, p.dtype) * d_lr))
    
                self.update_pre_conditioner(
                    p,
                    g,
                    step=step,
                    max_precondition_dim=self.max_precondition_dim,
                    merge_dims=self.merge_dims,
                    precondition_1d=self.precondition_1d,
                )
        
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
                "shampoo_beta": self.shampoo_beta,
                "precondition_frequency": self.precondition_frequency,
                "max_precondition_dim": self.max_precondition_dim,
                "merge_dims": self.merge_dims,
                "precondition_1d": self.precondition_1d,
                "correct_bias": self.correct_bias,
                "normalize_gradient": self.normalize_gradient,
                "data_format": self.data_format,
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