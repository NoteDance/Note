""" Muon
https://kellerjordan.github.io/posts/muon/

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import numpy as np
import math
import os


def zero_power_via_newton_schulz_5(G, steps):
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
    X = X / (tf.norm(X, axis=[-2, -1], keepdims=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = tf.matmul(X, tf.linalg.matrix_transpose(X))
        B = b * A + c * tf.matmul(A, A) # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + tf.matmul(B, X)
    
    if G.shape[-2] > G.shape[-1]:
        X = tf.linalg.matrix_transpose(X)
    return X


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


class Muon(optimizer.Optimizer):
    def __init__(
        self,
        params,
        learning_rate=2e-2,
        beta1=0.9,
        beta2=0.95,
        weight_decay=1e-2,
        momentum=0.95,
        weight_decouple=True,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_wd=0.0,
        adamw_eps=1e-8,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="muon",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_params = adamw_params
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        
        if adamw_params is not None:
            params.extend(adamw_params)
        self.params = params
            
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
    
    def set_muon_state(self, params, adamw_params):
        r"""Set use_muon flag."""
        for p in params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = len(p.shape) >= 2

        for p in adamw_params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = False
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum_buffer"
                                                    )
            self.moment1[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="moment1"
                                                    )
            self.moment2[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="moment2"
                                                    )
            self.use_muon[self._get_variable_index(var)] = None

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.moment1 = []
        self.moment2 = []
        self.use_muon = []
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum_buffer"
                                                    ))
            self.moment1.append(self.add_variable_from_reference(
                                reference_variable=var, name="moment1"
                                                    ))
            self.moment2.append(self.add_variable_from_reference(
                                reference_variable=var, name="moment2"
                                                    ))
            self.use_muon.append(None)
        self.set_muon_state(self.params, self.adamw_params)
    
    @staticmethod
    def adjust_lr_for_muon(lr, param_shape):
        adjusted_ratio = 0.2 * math.sqrt(max(param_shape[0], param_shape[1]))
        return lr * adjusted_ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        params = []
        for p, grad in zip(trainable_variables, grads):
            if self.use_muon[self._get_variable_index(p)]:
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon does not support sparse gradients')
                params.append(p)
        
        total_params = sum(np.prod(p.shape.as_list()) for p in params)
        updates_flat = tf.zeros(total_params, dtype=tf.bfloat16)
        curr_idx = 0
        
        for i, p in enumerate(params):
            if i % self.world_size != self.rank:
                curr_idx += np.prod(p.shape.as_list())
                continue

            grad = grads[self._get_variable_index(p)]
            if len(grad.shape) > 2:
                grad = tf.reshape(grad, (grad.shape[0], -1))

            buf = self.momentum_buffer[self._get_variable_index(p)]
            buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)

            grad = grad + self.momentum * (buf - grad) if self.nesterov else buf

            grad = tf.reshape(zero_power_via_newton_schulz_5(grad, num_steps=self.ns_steps), [-1])

            updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())] = grad  # fmt: skip
        
        if self.world_size > 1:  # pragma: no cover
            strategy = tf.distribute.get_strategy()
            updates_flat = strategy.reduce(tf.distribute.ReduceOp.SUM, updates_flat, axis=None)
        
        curr_idx: int = 0
        for p in params:
            g = tf.reshape(updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())], p.shape)  # fmt: skip

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * self.lr))
            elif self.weight_decay > 0.0:
                grads[self._get_variable_index(p)] += p * self.weight_decay

            lr = self.adjust_lr_for_muon(self.lr, p.shape) if self.use_adjusted_lr else self.lr

            p.assign_add(g * -lr * (max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5))
            curr_idx += np.prod(p.shape.as_list())

        params = [p for p in trainable_variables if not self.use_muon[self._get_variable_index(p)]]

        lr = self.adamw_lr_ratio * lr
        
        for p in params:
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
            step_size = lr / scale
            
            grad = grads[self._get_variable_index(p)]

            buf1 = self.moment1[self._get_variable_index(p)]
            buf2 = self.moment2[self._get_variable_index(p)]
            buf1.assign(buf1 + (1.0 - self.beta1) * (grad - buf1))
            buf2.assign(buf2 + (1.0 - self.beta2) * (tf.square(grad) - buf2))

            update = buf1 / tf.sqrt(buf2) + self.adamw_eps

            if self.weight_decouple:
                p.assign(p * (1.0 - self.adamw_wd * lr))
            elif self.adamw_wd > 0.0:
                grads[self._get_variable_index(p)] += p * self.adamw_wd

            p.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_wd": self.adamw_wd,
                "adamw_eps": self.adamw_eps,
                "world_size": self.world_size,
                "rank": self.rank,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class AdaMuon(optimizer.Optimizer):
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon: float = 1e-8,
        weight_decay=1e-2,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_params = None,
        adamw_betas = (0.9, 0.999),
        adamw_lr: float = 3e-4,
        adamw_wd: float = 0.0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamuon",
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
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_params = adamw_params
        self.adamw_betas = adamw_betas
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        
        if adamw_params is not None:
            params.extend(adamw_params)
        self.params = params
            
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
    
    def set_muon_state(self, params, adamw_params):
        r"""Set use_muon flag."""
        for p in params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = len(p.shape) >= 2

        for p in adamw_params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = False
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum_buffer"
                                                    )
            self.m[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="m"
                                                    )
            reshaped_var = tf.Variable(tf.reshape(var, (-1)))
            self.v[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=reshaped_var, name="v"
                                                    )
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.use_muon[self._get_variable_index(var)] = None

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.m = []
        self.v = []
        self.exp_avg = []
        self.exp_avg_sq = []
        self.use_muon = []
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum_buffer"
                                                    ))
            self.m.append(self.add_variable_from_reference(
                                reference_variable=var, name="m"
                                                    ))
            reshaped_var = tf.Variable(tf.reshape(var, (-1)))
            self.v.append(self.add_variable_from_reference(
                                reference_variable=reshaped_var, name="v"
                                                    ))
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.use_muon.append(None)
        self.set_muon_state(self.params, self.adamw_params)
    
    @staticmethod
    def get_adjusted_lr(lr: float, param_shape, use_adjusted_lr: bool = False) -> float:
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = math.prod(input_shape)

        ratio: float = (
            math.pow(max(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * math.sqrt(max(output_shape, input_shape))
        )

        return lr * ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        params = []
        for p, grad in zip(trainable_variables, grads):
            if self.use_muon[self._get_variable_index(p)]:
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon does not support sparse gradients')
                params.append(p)
        
        total_params = sum(np.prod(p.shape.as_list()) for p in params)
        updates_flat = tf.zeros(total_params, dtype=tf.bfloat16)
        curr_idx = 0
        
        for i, p in enumerate(params):
            if i % self.world_size != self.rank:
                curr_idx += np.prod(p.shape.as_list())
                continue

            grad = grads[self._get_variable_index(p)]
            if len(grad.shape) > 2:
                grad = tf.reshape(grad, (grad.shape[0], -1))

            buf = self.momentum_buffer[self._get_variable_index(p)]
            buf.assign(buf + 1.0 - self.beta1 * (grad - buf))

            grad = grad + self.beta1 * (buf - grad) if self.nesterov else buf

            grad = tf.reshape(zero_power_via_newton_schulz_5(grad, num_steps=self.ns_steps), [-1])

            updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())] = grad  # fmt: skip
        
        if self.world_size > 1:  # pragma: no cover
            strategy = tf.distribute.get_strategy()
            updates_flat = strategy.reduce(tf.distribute.ReduceOp.SUM, updates_flat, axis=None)
        
        curr_idx: int = 0
        for i, p in enumerate(params):
            if i % self.world_size != self.rank:
                continue
            
            g = updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())]  # fmt: skip
            
            v = self.v[self._get_variable_index(p)]
            v.assign(v * self.beta2 + g * g * (1.0 - self.beta2))
            
            step = tf.cast(self.iterations + 1, p.dtype)
            bias_correction2 = 1 - self.beta2 ** step
            
            update = g / tf.sqrt(v / bias_correction2) + self.epsilon
            update = tf.reshape(update, p.shape)
            
            update = update * 0.2 * math.sqrt(np.prod(p.shape.as_list())) / tf.norm(update) + self.epsilon

            lr = self.get_adjusted_lr(self.lr, p.shape, self.use_adjusted_lr)

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * self.lr))
            elif self.weight_decay > 0.0:
                grads[self._get_variable_index(p)] = tf.reshape(grads[self._get_variable_index(p)], p.shape)
                grads[self._get_variable_index(p)] += p * self.weight_decay
            
            lr = tf.cast(lr, p.dtype)
            p.assign_add(-lr * update)
            curr_idx += np.prod(p.shape.as_list())

        params = [p for p in trainable_variables if not self.use_muon[self._get_variable_index(p)]]

        lr = self.adamw_lr_ratio * lr
        
        for p in params:
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
            step_size = lr / scale
            
            grad = grads[self._get_variable_index(p)]

            buf1 = self.exp_avg[self._get_variable_index(p)]
            buf2 = self.exp_avg_sq[self._get_variable_index(p)]
            buf1.assign(buf1 + (1.0 - self.beta1) * (grad - buf1))
            buf2.assign(buf2 + (1.0 - self.beta2) * (tf.square(grad) - buf2))

            update = buf1 / tf.sqrt(buf2) + self.epsilon

            if self.weight_decouple:
                p.assign(p * (1.0 - self.adamw_wd * lr))

            p.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_betas": self.adamw_betas,
                "adamw_lr": self.adamw_lr,
                "adamw_wd": self.adamw_wd,
                "world_size": self.world_size,
                "rank": self.rank,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class Muon_sn(optimizer.Optimizer):
    def __init__(
        self,
        params,
        learning_rate=2e-2,
        beta1=0.9,
        beta2=0.95,
        weight_decay=1e-2,
        momentum=0.95,
        weight_decouple=True,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_wd=0.0,
        adamw_eps=1e-8,
        sn=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="muon_sn",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_params = adamw_params
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.sn = sn
        
        if adamw_params is not None:
            params.extend(adamw_params)
        self.params = params
            
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
    
    def set_muon_state(self, params, adamw_params):
        r"""Set use_muon flag."""
        for p in params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = len(p.shape) >= 2

        for p in adamw_params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = False
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum_buffer"
                                                    )
            self.moment1[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="moment1"
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
                self.moment2 = self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="moment2"
                    )
            else:
                self.moment2 = self.add_variable_from_reference(
                                    reference_variable=var, name="moment2"
                                                        )
            self.use_muon[self._get_variable_index(var)] = None

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.moment1 = []
        self.moment2 = []
        self.use_muon = []
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum_buffer"
                                                    ))
            self.moment1.append(self.add_variable_from_reference(
                                reference_variable=var, name="moment1"
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
                self.moment2.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="moment2"
                    ))
            else:
                self.moment2.append(self.add_variable_from_reference(
                                    reference_variable=var, name="moment2"
                                                        ))
            self.use_muon.append(None)
        self.set_muon_state(self.params, self.adamw_params)
    
    @staticmethod
    def adjust_lr_for_muon(lr, param_shape):
        adjusted_ratio = 0.2 * math.sqrt(max(param_shape[0], param_shape[1]))
        return lr * adjusted_ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        params = []
        for p, grad in zip(trainable_variables, grads):
            if self.use_muon[self._get_variable_index(p)]:
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon does not support sparse gradients')
                params.append(p)
        
        total_params = sum(np.prod(p.shape.as_list()) for p in params)
        updates_flat = tf.zeros(total_params, dtype=tf.bfloat16)
        curr_idx = 0
        
        for i, p in enumerate(params):
            if i % self.world_size != self.rank:
                curr_idx += np.prod(p.shape.as_list())
                continue

            grad = grads[self._get_variable_index(p)]
            if len(grad.shape) > 2:
                grad = tf.reshape(grad, (grad.shape[0], -1))

            buf = self.momentum_buffer[self._get_variable_index(p)]
            buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)

            grad = grad + self.momentum * (buf - grad) if self.nesterov else buf

            grad = tf.reshape(zero_power_via_newton_schulz_5(grad, num_steps=self.ns_steps), [-1])

            updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())] = grad  # fmt: skip
        
        if self.world_size > 1:  # pragma: no cover
            strategy = tf.distribute.get_strategy()
            updates_flat = strategy.reduce(tf.distribute.ReduceOp.SUM, updates_flat, axis=None)
        
        curr_idx: int = 0
        for p in params:
            g = tf.reshape(updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())], p.shape)  # fmt: skip

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * self.lr))
            elif self.weight_decay > 0.0:
                grads[self._get_variable_index(p)] += p * self.weight_decay

            lr = self.adjust_lr_for_muon(self.lr, p.shape) if self.use_adjusted_lr else self.lr

            p.assign_add(g * -lr * (max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5))
            curr_idx += np.prod(p.shape.as_list())

        params = [p for p in trainable_variables if not self.use_muon[self._get_variable_index(p)]]

        lr = self.adamw_lr_ratio * lr
        
        for p in params:
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
            step_size = lr / scale
            
            grad = grads[self._get_variable_index(p)]
            size = tf.size(grad)

            buf1 = self.moment1[self._get_variable_index(p)]
            buf2 = self.moment2[self._get_variable_index(p)]
            if self.sn:
                reshaped_grad = tf.reshape(grad, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
            else:
                second_moment_update = tf.pow(grad, 2)
            buf1.assign(buf1 + (1.0 - self.beta1) * (grad - buf1))
            buf2.assign(buf2 + (1.0 - self.beta2) * (second_moment_update - buf2))

            if self.sn:
                buf1.assign(buf1 * self.beta1 + grad * (1.0 - self.beta1))
                numerator = tf.reshape(buf1, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                normed_grad = tf.reshape((numerator / tf.sqrt(buf2) + self.adamw_eps), p.shape)
                update = normed_grad
            else:
                update = buf1 / tf.sqrt(buf2) + self.adamw_eps

            if self.weight_decouple:
                p.assign(p * (1.0 - self.adamw_wd * lr))
            elif self.adamw_wd > 0.0:
                grads[self._get_variable_index(p)] += p * self.adamw_wd

            p.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_wd": self.adamw_wd,
                "adamw_eps": self.adamw_eps,
                "sn": self.sn,
                "world_size": self.world_size,
                "rank": self.rank,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class AdaMuon_sn(optimizer.Optimizer):
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon: float = 1e-8,
        weight_decay=1e-2,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_params = None,
        adamw_betas = (0.9, 0.999),
        adamw_lr: float = 3e-4,
        adamw_wd: float = 0.0,
        sn=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamuon_sn",
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
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_params = adamw_params
        self.adamw_betas = adamw_betas
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.sn = sn
        
        if adamw_params is not None:
            params.extend(adamw_params)
        self.params = params
            
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
    
    def set_muon_state(self, params, adamw_params):
        r"""Set use_muon flag."""
        for p in params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = len(p.shape) >= 2

        for p in adamw_params:
            if p.trainable:
                self.use_muon[self._get_variable_index(p)] = False
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum_buffer"
                                                    )
            self.m[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="m"
                                                    )
            reshaped_var = tf.Variable(tf.reshape(var, (-1)))
            self.v[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=reshaped_var, name="v"
                                                    )
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
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
                self.exp_avg_sq = self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="exp_avg_sq"
                    )
            else:
                self.exp_avg_sq = self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        )
            self.use_muon[self._get_variable_index(var)] = None

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.m = []
        self.v = []
        self.exp_avg = []
        self.exp_avg_sq = []
        self.use_muon = []
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum_buffer"
                                                    ))
            self.m.append(self.add_variable_from_reference(
                                reference_variable=var, name="m"
                                                    ))
            reshaped_var = tf.Variable(tf.reshape(var, (-1)))
            self.v.append(self.add_variable_from_reference(
                                reference_variable=reshaped_var, name="v"
                                                    ))
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
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="exp_avg_sq"
                    ))
            else:
                self.exp_avg_sq.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        ))
            self.use_muon.append(None)
        self.set_muon_state(self.params, self.adamw_params)
    
    @staticmethod
    def get_adjusted_lr(lr: float, param_shape, use_adjusted_lr: bool = False) -> float:
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = math.prod(input_shape)

        ratio: float = (
            math.pow(max(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * math.sqrt(max(output_shape, input_shape))
        )

        return lr * ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        params = []
        for p, grad in zip(trainable_variables, grads):
            if self.use_muon[self._get_variable_index(p)]:
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon does not support sparse gradients')
                params.append(p)
        
        total_params = sum(np.prod(p.shape.as_list()) for p in params)
        updates_flat = tf.zeros(total_params, dtype=tf.bfloat16)
        curr_idx = 0
        
        for i, p in enumerate(params):
            if i % self.world_size != self.rank:
                curr_idx += np.prod(p.shape.as_list())
                continue

            grad = grads[self._get_variable_index(p)]
            if len(grad.shape) > 2:
                grad = tf.reshape(grad, (grad.shape[0], -1))

            buf = self.momentum_buffer[self._get_variable_index(p)]
            buf.assign(buf + 1.0 - self.beta1 * (grad - buf))

            grad = grad + self.beta1 * (buf - grad) if self.nesterov else buf

            grad = tf.reshape(zero_power_via_newton_schulz_5(grad, num_steps=self.ns_steps), [-1])

            updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())] = grad  # fmt: skip
        
        if self.world_size > 1:  # pragma: no cover
            strategy = tf.distribute.get_strategy()
            updates_flat = strategy.reduce(tf.distribute.ReduceOp.SUM, updates_flat, axis=None)
        
        curr_idx: int = 0
        for i, p in enumerate(params):
            if i % self.world_size != self.rank:
                continue
            
            g = updates_flat[curr_idx:curr_idx + np.prod(p.shape.as_list())]  # fmt: skip
            
            v = self.v[self._get_variable_index(p)]
            v.assign(v * self.beta2 + g * g * (1.0 - self.beta2))
            
            step = tf.cast(self.iterations + 1, p.dtype)
            bias_correction2 = 1 - self.beta2 ** step
            
            update = g / tf.sqrt(v / bias_correction2) + self.epsilon
            update = tf.reshape(update, p.shape)
            
            update = update * 0.2 * math.sqrt(np.prod(p.shape.as_list())) / tf.norm(update) + self.epsilon

            lr = self.get_adjusted_lr(self.lr, p.shape, self.use_adjusted_lr)

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * self.lr))
            elif self.weight_decay > 0.0:
                grads[self._get_variable_index(p)] = tf.reshape(grads[self._get_variable_index(p)], p.shape)
                grads[self._get_variable_index(p)] += p * self.weight_decay
            
            lr = tf.cast(lr, p.dtype)
            p.assign_add(-lr * update)
            curr_idx += np.prod(p.shape.as_list())

        params = [p for p in trainable_variables if not self.use_muon[self._get_variable_index(p)]]

        lr = self.adamw_lr_ratio * lr
        
        for p in params:
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
            step_size = lr / scale
            
            grad = grads[self._get_variable_index(p)]
            size = tf.size(grad)

            buf1 = self.exp_avg[self._get_variable_index(p)]
            buf2 = self.exp_avg_sq[self._get_variable_index(p)]
            if self.sn:
                reshaped_grad = tf.reshape(grad, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
            else:
                second_moment_update = tf.pow(grad, 2)
            buf1.assign(buf1 + (1.0 - self.beta1) * (grad - buf1))
            buf2.assign(buf2 + (1.0 - self.beta2) * (second_moment_update - buf2))

            if self.sn:
                buf1.assign(buf1 * self.beta1 + grad * (1.0 - self.beta1))
                numerator = tf.reshape(buf1, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                normed_grad = tf.reshape((numerator / tf.sqrt(buf2) + self.epsilon), p.shape)
                update = normed_grad
            else:
                update = buf1 / tf.sqrt(buf2) + self.epsilon

            if self.weight_decouple:
                p.assign(p * (1.0 - self.adamw_wd * lr))

            p.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_betas": self.adamw_betas,
                "adamw_lr": self.adamw_lr,
                "adamw_wd": self.adamw_wd,
                "sn": self.sn,
                "world_size": self.world_size,
                "rank": self.rank,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
