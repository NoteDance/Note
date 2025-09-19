""" Muon
https://kellerjordan.github.io/posts/muon/

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import numpy as np
import math


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


class Muon(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=2e-2,
        beta1=0.9,
        beta2=0.95,
        weight_decay=1e-2,
        momentum=0.95,
        weight_decouple=True,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_wd=0.0,
        adamw_eps=1e-8,
        use_muon=True,
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.use_muon = use_muon

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.moment1 = []
        self.moment2 = []
        for var in var_list:
            if self.use_muon:
                self.momentum_buffer.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                                        ))
            else:
                self.moment1.append(self.add_variable_from_reference(
                    reference_variable=var, name="moment1"
                                        ))
                self.moment2.append(self.add_variable_from_reference(
                                    reference_variable=var, name="moment2"
                                                        ))
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.use_muon:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon does not support sparse gradients')
                
                lr = tf.cast(learning_rate, p.dtype)
                
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.weight_decay * lr))
                elif self.weight_decay > 0.0:
                    grad += p * self.weight_decay
                
                buf = self.momentum_buffer[self._get_variable_index(p)]
                buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)
                
                update = grad + self.momentum * (buf - grad) if self.nesterov else buf
                
                if len(update.shape) > 2:
                    update = tf.reshape(update, (len(update), -1))
                
                update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
    
                lr = self.get_adjusted_lr(lr, p.shape, self.use_adjusted_lr)
    
                p.assign_add(tf.reshape(update, p.shape) * -lr)
        else:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon does not support sparse gradients')
                    
                lr = tf.cast(self.adamw_lr, p.dtype)
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2 = 1 - self.beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
                step_size = lr / scale
    
                buf1 = self.moment1[self._get_variable_index(p)]
                buf2 = self.moment2[self._get_variable_index(p)]
                buf1.assign(buf1 + (1.0 - self.beta1) * (grad - buf1))
                buf2.assign(buf2 + (1.0 - self.beta2) * (tf.square(grad) - buf2))
    
                update = buf1 / (tf.sqrt(buf2) + self.adamw_eps)
    
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.adamw_wd * lr))
                elif self.adamw_wd > 0.0:
                    grads[self._get_variable_index(p)] += p * self.adamw_wd
    
                p.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
                "use_muon": self.use_muon,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class DistributedMuon(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=2e-2,
        weight_decay=0.0,
        momentum=0.95,
        weight_decouple=True,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        use_muon = True,
        cautious = False,
        maximize: bool = False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="distributedmuon",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.use_muon = use_muon
        self.cautious = cautious
        self.maximize = maximize
            
        self.world_size = tf.distribute.get_strategy().num_replicas_in_sync
        self.rank = int(tf.distribute.get_replica_context().replica_id_in_sync_group)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.exp_avg = []
        self.exp_avg_sq = []
        for var in var_list:
            if var.trainable and len(var.shape) >= 2:
                self.momentum_buffer.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                                        ))
                self.exp_avg.append(None)
                self.exp_avg_sq.append(None)
            else:
                self.exp_avg.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg"
                                                        ))
                self.exp_avg_sq.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        ))
                self.momentum_buffer.append(None)
        if self.use_muon:
            self.padded_params = var_list + [self.add_variable_from_reference(
                                reference_variable=var_list[-1])] * (
                self.world_size - len(var_list) % self.world_size
            )
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
    def distributed_step(self, padded_params):
        strategy = tf.distribute.get_strategy()
        
        def replica_fn(padded_params):
            rc = tf.distribute.get_replica_context()
    
            new_padded = list(padded_params)
            for i in range(0, len(padded_params), self.world_size):
                local = padded_params[i + self.rank]
    
                local_expanded = tf.expand_dims(local, axis=0)  # shape [1, ...]
                gathered = rc.all_gather(local_expanded, axis=0)  # shape [world_size, ...]
                per_rank_list = tf.unstack(gathered, num=self.world_size, axis=0)
                
                for j in range(self.world_size):
                    new_padded[i : i + j].assign(per_rank_list[j])
    
            return new_padded
    
        return strategy.run(replica_fn, args=(padded_params,))
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.use_muon:
            for i in range(len(trainable_variables))[:: self.world_size]:
                grad = grads[self._get_variable_index(trainable_variables[i])]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'DistributedMuon does not support sparse gradients')
                if i + self.rank < len(trainable_variables):
                    p = trainable_variables[i + self.rank]
                    
                    lr = tf.cast(learning_rate, p.dtype)
    
                    if self.maximize:
                        grad = -grad
    
                    if self.weight_decouple:
                        trainable_variables[i].assign(trainable_variables[i] * (1.0 - self.weight_decay * lr))
                    elif self.weight_decay > 0.0:
                        grad += trainable_variables[i] * self.weight_decay
    
                    buf = self.momentum_buffer[self._get_variable_index(trainable_variables[i])]
                    buf.assign(buf + (1.0 - self.momentum) * (grad - buf))
    
                    update = grad + self.momentum * (buf - grad) if self.nesterov else buf
                    if len(update.shape) > 2:
                        update = tf.reshape(update, (len(update), -1))
    
                    update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
    
                    if self.cautions:
                        mask = tf.cast(update * grad > 0, grad.dtype)
                        mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                        update *= mask
    
                    lr = self.get_adjusted_lr(lr, p.shape, use_adjusted_lr=self.use_adjusted_lr)
    
                    trainable_variables[i].assign_add(tf.reshape(update, (p.shape)) * -lr)
    
                self.distributed_step(self.padded_params)
        else:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'DistributedMuon does not support sparse gradients')
                lr = tf.cast(self.adamw_lr, p.dtype)
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                grad = grads[self._get_variable_index(p)]
    
                exp_avg = self.exp_avg[self._get_variable_index(p)]
                exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
    
                beta1, beta2 = self.adamw_betas
    
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
    
                exp_avg.assign(beta1 * exp_avg + (1.0 - beta1) * grad)
                exp_avg_sq.assign(beta2 * exp_avg_sq + (1.0 - beta2) * tf.square(grad))
    
                de_nom = (tf.sqrt(exp_avg_sq) + self.adamw_eps) / bias_correction2_sq
    
                p.assign_add(-lr * (exp_avg / bias_correction1) / de_nom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_betas": self.adamw_betas,
                "adamw_wd": self.adamw_wd,
                "adamw_eps": self.adamw_eps,
                "use_muon": self.use_muon,
                "cautious": self.cautious,
                "maximize": self.maximize,
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
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon: float = 1e-8,
        weight_decay=1e-2,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_betas = (0.9, 0.999),
        adamw_lr: float = 3e-4,
        adamw_wd: float = 0.0,
        use_muon: bool = True,
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_betas = adamw_betas
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.use_muon = use_muon

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.m = []
        self.v = []
        self.exp_avg = []
        self.exp_avg_sq = []
        for var in var_list:
            if self.use_muon:
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
            else:
                self.exp_avg.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg"
                                                        ))
                self.exp_avg_sq.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        ))
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.use_muon:
            for i, p in enumerate(trainable_variables):
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        ' AdaMuon does not support sparse gradients')
                
                lr = tf.cast(learning_rate, p.dtype)
                    
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.weight_decay * lr))
                elif self.weight_decay > 0.0:
                    grad += p * self.weight_decay
                
                buf = self.momentum_buffer[self._get_variable_index(p)]
                buf.assign(buf + 1.0 - self.beta1 * (grad - buf))
                
                update = grad + self.beta1 * (buf - grad) if self.nesterov else buf
                
                if len(update.shape) > 2:
                    update = tf.reshape(update, (len(update), -1))
                
                update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
                
                v = self.v[self._get_variable_index(p)]
                v.assign(v * self.beta2 + update * update * (1.0 - self.beta2))
                
                step = tf.cast(self.iterations + 1, p.dtype)
                bias_correction2 = 1 - self.beta2 ** step
                
                update = update / tf.sqrt(v / bias_correction2) + self.epsilon
                update = tf.reshape(update, p.shape)
                
                update = update * 0.2 * math.sqrt(np.prod(p.shape.as_list())) / tf.norm(update) + self.epsilon
    
                lr = self.get_adjusted_lr(lr, p.shape, self.use_adjusted_lr)
                
                p.assign_add(-lr * update)
        else:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        ' AdaMuon does not support sparse gradients')
                lr = tf.cast(self.adamw_lr, p.dtype)
                
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
    
                update = buf1 / (tf.sqrt(buf2) + self.epsilon)
    
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.adamw_wd * lr))
    
                p.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
                "use_muon": self.use_muon,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass

    
class AdaGO(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=5e-2,
        epsilon=5e-4,
        weight_decay=0.0,
        momentum=0.95,
        weight_decouple=True,
        gamma=10.0,
        v=1e-6,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_betas=(0.9,0.95),
        adamw_wd=0.0,
        adamw_eps=1e-10,
        maximize=False,
        cautious=False,
        use_muon=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adago",
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
        self.epsilon = epsilon
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.gamma = gamma
        self.v = v
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.cautious = cautious
        self.maximize = maximize
        self.use_muon = use_muon

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.v_ = []
        self.exp_avg = []
        self.exp_avg_sq = []
        for var in var_list:
            if self.use_muon:
                self.momentum_buffer.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                                        ))
                self.v_.append(tf.Variable(self.v, dtype=var.dtype))
                self._track_variable(self.v_[-1])
            else:
                self.exp_avg.append(self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                                        ))
                self.exp_avg_sq.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg_sq"
                                                        ))
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.use_muon:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'AdaGO does not support sparse gradients')
                
                lr = tf.cast(learning_rate, p.dtype)
                
                if self.maximize:
                    grad = -grad
                
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.weight_decay * lr))
                elif self.weight_decay > 0.0:
                    grad += p * self.weight_decay
                
                buf = self.momentum_buffer[self._get_variable_index(p)]
                v = self.v_[self._get_variable_index(p)]
                buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)
                
                v.assign_add(tf.minimum(tf.pow(tf.norm(grad, ord=2.0), 2), self.gamma ** 2))
                
                update = grad + self.momentum * (buf - grad) if self.nesterov else buf
                
                if len(update.shape) > 2:
                    update = tf.reshape(update, (len(update), -1))
                
                update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
    
                lr = self.get_adjusted_lr(lr, p.shape, self.use_adjusted_lr)
    
                p.assign_add(tf.reshape(update, p.shape) * -tf.maximum(self.epsilon, lr * tf.minimum(tf.norm(grad, ord=2), self.gamma) / v))
        else:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'AdaGO does not support sparse gradients')
                
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.adamw_wd * lr))
                elif self.adamw_wd > 0.0:
                    grads[self._get_variable_index(p)] += p * self.adamw_wd
                    
                lr = tf.cast(self.adamw_lr, p.dtype)
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                beta1, beta2 = self.adamw_betas
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction2_sq = tf.sqrt(bias_correction2)
    
                exp_avg = self.exp_avg[self._get_variable_index(p)]
                exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                exp_avg.assign(exp_avg * beta1 + (1.0 - beta1) * grad)
                exp_avg_sq.assign(exp_avg_sq * beta2 + (1.0 - beta2) * tf.square(grad))
                
                de_nom = (tf.sqrt(exp_avg_sq) + self.adamw_eps) / bias_correction2_sq
    
                p.assign_add(-lr * (exp_avg / bias_correction1) / de_nom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "gamma": self.gamma,
                "v": self.v,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_betas": self.adamw_betas,
                "adamw_wd": self.adamw_wd,
                "adamw_eps": self.adamw_eps,
                "cautious": self.cautious,
                "maximize": self.maximize,
                "use_muon": self.use_muon,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class Muon_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=2e-2,
        beta1=0.9,
        beta2=0.95,
        beta3=0.9999,
        weight_decay=1e-2,
        momentum=0.95,
        weight_decouple=True,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_wd=0.0,
        adamw_eps=1e-8,
        use_muon=True,
        subset_size=-1,
        sn=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        agc=True,
        cautious=True,
        aem=False,
        alpha=5.0,
        t_alpha_beta3=None,
        sophia=True,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
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
        name="muon_e",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.use_muon = use_muon
        self.subset_size = subset_size
        self.sn = sn
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.aem = aem
        self.alpha = alpha
        self.t_alpha_beta3 = t_alpha_beta3
        self.sophia = sophia
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.moment1 = []
        self.moment1_slow = []
        self.moment2 = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        self.hessian_moment = []
        self.hessian = []
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
            if self.use_muon:
                if not self.pnm:
                    self.momentum_buffer.append(self.add_variable_from_reference(
                        reference_variable=var, name="momentum_buffer"
                                            ))
                if self.trust_ratio and self.sn:
                    size = tf.size(var)
                    
                    def true_fn():
                        return self.subset_size
                    def false_fn():
                        return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                    self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                        size,
                        tf.cond(self.subset_size > 0, true_fn, false_fn)
                    ))
            else:
                if not self.pnm:
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
                if self.aem:
                    self.moment1_slow.append(self.add_variable_from_reference(
                        reference_variable=var, name="moment1_slow"
                                            ))
                if self.sophia:
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=var, name="hessian_moment"
                                            ))
                    self.hessian.append(self.add_variable_from_reference(
                                        reference_variable=var, name="hessian"
                                                            ))
                if self.DAdapt:
                    self.s.append(self.add_variable_from_reference(
                        reference_variable=var, name="s"
                                            ))
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
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
    
    def compute_hutchinson_hessian(
        self,
        params,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params = [p for p in params if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [grads[self._get_variable_index(p)] for p in params if not tf.keras.backend.is_sparse(grads[self._get_variable_index(p)])]

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
    
    def apply_gradients(self, grads_and_vars, tape=None):
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
        if self.use_muon:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon_e does not support sparse gradients')
                
                lr = tf.cast(learning_rate, p.dtype)
                
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.weight_decay * lr))
                elif self.weight_decay > 0.0:
                    grad += p * self.weight_decay
                
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad) 
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                if not self.pnm:
                    buf = self.momentum_buffer[self._get_variable_index(p)]
                    buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)
                else:
                    noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    pos_momentum.assign(pos_momentum * self.beta1 ** 2 + grad * (1.0 - self.beta1 ** 2))
                    buf = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
    
                update = grad + self.momentum * (buf - grad) if self.nesterov else buf
                
                if len(grad.shape) > 2:
                    update = tf.reshape(update, (len(update), -1))
    
                update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
    
                lr = self.get_adjusted_lr(lr, p.shape, self.use_adjusted_lr)
                
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                    numel = tf.cast(tf.size(mask), grad.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    if self.sn:
                        size = tf.size(p)
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
    
                p.assign_add(tf.reshape(update, p.shape) * -lr)
                
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
        else:
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
            
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'Muon_e does not support sparse gradients')
                
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad) 
                    
                lr = tf.cast(self.adamw_lr, p.dtype)
                step = tf.cast(self.iterations + 1, p.dtype)
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2 = 1 - self.beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
                step_size = lr / scale
                if self.DAdapt:
                    d_lr = self.d0 * self.adamw_lr
                    d_lr = d_lr / scale
                    d_lr = tf.cast(d_lr, p.dtype)
                    s = self.s[self._get_variable_index(p)]
                
                if self.aem:
                    beta1 = tf.cast(self.beta1, p.dtype)
                    beta3 = tf.cast(self.beta3, p.dtype)
                    
                    alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                    beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
                    
                    moment1_slow = self.moment1_slow[self._get_variable_index(p)]
                    
                    clip = tf.pow(step, 0.25)
                
                size = tf.size(grad)
    
                if not self.pnm:
                    buf1 = self.moment1[self._get_variable_index(p)]
                if self.sophia:
                    hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                else:
                    buf2 = self.moment2[self._get_variable_index(p)]
                if not self.aem:
                    normed_grad = grad
                else:
                    normed_grad = tf.clip_by_value(
                        grad / tf.maximum(tf.sqrt(buf2), self.adamw_eps if self.adamw_eps is not None else 1e-8),
                        clip_value_min=-clip,
                        clip_value_max= clip,
                    )
                if self.sn:
                    reshaped_grad = tf.reshape(grad, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                else:
                    second_moment_update = tf.pow(grad, 2)
                if not self.pnm:
                    if self.DAdapt:
                        beta2_sq = math.sqrt(self.beta2)
                        buf1.assign(buf1 + (1.0 - self.beta1) * (normed_grad * d_lr - buf1))
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        buf1.assign(buf1 + (1.0 - self.beta1) * (normed_grad - buf1))
                else:
                    noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    if self.DAdapt:
                        beta2_sq = math.sqrt(self.beta2)
                        pos_momentum.assign(pos_momentum * (1.0 - self.beta1**2) * (normed_grad * d_lr - pos_momentum))
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        pos_momentum.assign(pos_momentum * (1.0 - self.beta1**2) * (normed_grad - pos_momentum))
                    buf1 = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                if self.sophia:
                    def true_fn2():
                        hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))
                    def false_fn2():
                        pass
                    tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
                else:
                    buf2.assign(buf2 + (1.0 - self.beta2) * (second_moment_update - buf2))
                
                if self.aem:
                    moment1_slow.assign(moment1_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                    
                if self.sophia:
                    de_nom = tf.maximum(hessian_moment, self.adamw_eps)
                else:
                    de_nom = tf.sqrt(buf2) + self.adamw_eps
                
                if self.DAdapt:
                    flat_grad = tf.reshape(grad, [-1])
                    flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                    dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                    self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
    
                if not self.DAdapt:
                    if self.aem:
                        buf1 += moment1_slow * alpha_t
                    if self.sn:
                        numerator = tf.reshape(buf1, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        normed_grad = tf.reshape(numerator / de_nom, p.shape)
                        update = normed_grad
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                    else:
                        normed_grad = buf1 / de_nom
                        update = normed_grad
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                    if self.cautious:
                        mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                        numel = tf.cast(tf.size(mask), grad.dtype)
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
        
                    if self.weight_decouple:
                        p.assign(p * (1.0 - self.adamw_wd * lr))
                    elif self.adamw_wd > 0.0:
                        grads[self._get_variable_index(p)] += p * self.adamw_wd
        
                    p.assign_add(update * -step_size)
                    
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
                    step = self.iterations + 1
                    bias_correction1 = 1 - self.beta1 ** step
                    bias_correction2 = 1 - self.beta2 ** step
                    scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
                    d_lr = self.d0 * self.adamw_lr
                    
                    beta2_sq = math.sqrt(self.beta2)
                    
                    d = self.d0_
                    self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                    
                    if self.lr > 0.0:
                        d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                        d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                    
                    self.d0_.assign(d)
                    
                    for p in trainable_variables:
                        d_lr = tf.cast(d_lr, p.dtype)
                        
                        step = tf.cast(self.iterations + 1, p.dtype)
                        
                        grad = grads[self._get_variable_index(p)]
                        
                        if self.weight_decouple:
                            p.assign(p * (1.0 - self.adamw_wd * d_lr))
                        elif self.adamw_wd > 0.0:
                            grad += p * self.adamw_wd
                        
                        if not self.pnm:
                            buf1 = self.moment1[self._get_variable_index(p)]
                        else:
                            noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                            def true_fn():
                                return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                            def false_fn():
                                return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                            pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                            buf1 = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                        
                        if self.sophia:
                            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                        else:
                            buf2 = self.moment2[self._get_variable_index(p)]
                        
                        if self.aem:
                            moment1_slow = self.moment1_slow[self._get_variable_index(p)]
                            buf1 += moment1_slow * alpha_t
                        
                        if self.sophia:
                            de_nom = tf.maximum(hessian_moment, self.adamw_eps)
                        else:
                            de_nom = tf.sqrt(buf2) + self.adamw_eps
                            
                        if self.sn:
                            numerator = tf.reshape(buf1, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                            normed_grad = tf.reshape(numerator / de_nom, p.shape)
                            update = normed_grad
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        else:
                            normed_grad = buf1 / de_nom
                            update = normed_grad
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        if self.cautious:
                            mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                            numel = tf.cast(tf.size(mask), grad.dtype)
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
                        
                        step_size = d_lr / scale
            
                        p.assign_add(update * -step_size)
                        
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
                "use_muon": self.use_muon,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "cautious": self.cautious,
                "aem": self.aem,
                "alpha": self.alpha,
                "t_alpha_beta3": self.t_alpha_beta3,
                "sophia": self.sophia,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
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


class DistributedMuon_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=2e-2,
        weight_decay=0.0,
        momentum=0.95,
        weight_decouple=True,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_betas = (0.9, 0.95, 0.9999),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        use_muon = True,
        cautious = False,
        subset_size=-1,
        sn=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        agc=True,
        aem=False,
        alpha=5.0,
        t_alpha_beta3=None,
        sophia=True,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        d0=1e-6,
        growth_rate=float('inf'),
        DAdapt=True,
        trust_ratio=False,
        trust_clip=False,
        maximize: bool = False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="distributedmuon_e",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.use_muon = use_muon
        self.cautious = cautious
        self.subset_size = subset_size
        self.sn = sn
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.aem = aem
        self.alpha = alpha
        self.t_alpha_beta3 = t_alpha_beta3
        self.sophia = sophia
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.maximize = maximize
            
        self.world_size = tf.distribute.get_strategy().num_replicas_in_sync
        self.rank = int(tf.distribute.get_replica_context().replica_id_in_sync_group)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.exp_avg = []
        self.exp_avg_slow = []
        self.exp_avg_sq = []
        self.subset_size_ = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.hessian_moment = []
        self.hessian = []
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
            if var.trainable and len(var.shape) >= 2:
                if not self.pnm:
                    self.momentum_buffer.append(self.add_variable_from_reference(
                        reference_variable=var, name="momentum_buffer"
                                            ))
                    self.exp_avg.append(None)
                self.exp_avg_sq.append(None)
                if self.aem:
                    self.exp_avg_slow.append(None)
                if self.sophia:
                    self.hessian_moment.append(None)
                    self.hessian.append(None)
                if self.DAdapt:
                    self.s.append(None)
                if self.trust_ratio and self.sn:
                    size = tf.size(var)
                    
                    def true_fn():
                        return self.subset_size
                    def false_fn():
                        return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                    self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                        size,
                        tf.cond(self.subset_size > 0, true_fn, false_fn)
                    ))
            else:
                if not self.pnm:
                    self.exp_avg.append(self.add_variable_from_reference(
                                        reference_variable=var, name="exp_avg"
                                                            ))
                    self.momentum_buffer.append(None)
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
                if self.aem:
                    self.exp_avg_slow.append(self.add_variable_from_reference(
                        reference_variable=var, name="moment1_slow"
                                            ))
                if self.sophia:
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=var, name="hessian_moment"
                                            ))
                    self.hessian.append(self.add_variable_from_reference(
                                        reference_variable=var, name="hessian"
                                                            ))
                if self.DAdapt:
                    self.s.append(self.add_variable_from_reference(
                        reference_variable=var, name="s"
                                            ))
          
        if self.use_muon:
            self.padded_params = var_list + [self.add_variable_from_reference(
                                reference_variable=var_list[-1])] * (
                self.world_size - len(var_list) % self.world_size
            )
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
    def distributed_step(self, padded_params):
        strategy = tf.distribute.get_strategy()
        
        def replica_fn(padded_params):
            rc = tf.distribute.get_replica_context()
    
            new_padded = list(padded_params)
            for i in range(0, len(padded_params), self.world_size):
                local = padded_params[i + self.rank]
    
                local_expanded = tf.expand_dims(local, axis=0)  # shape [1, ...]
                gathered = rc.all_gather(local_expanded, axis=0)  # shape [world_size, ...]
                per_rank_list = tf.unstack(gathered, num=self.world_size, axis=0)
                
                for j in range(self.world_size):
                    new_padded[i : i + j].assign(per_rank_list[j])
    
            return new_padded
    
        return strategy.run(replica_fn, args=(padded_params,))
    
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
    
    def compute_hutchinson_hessian(
        self,
        params,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params = [p for p in params if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [grads[self._get_variable_index(p)] for p in params if not tf.keras.backend.is_sparse(grads[self._get_variable_index(p)])]

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
    
    def apply_gradients(self, grads_and_vars, tape=None):
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
        if self.use_muon:
            for i in range(len(trainable_variables))[:: self.world_size]:
                grad = grads[self._get_variable_index(trainable_variables[i])]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'DistributedMuon_e does not support sparse gradients')
                
                if self.agc:
                    grads[self._get_variable_index(trainable_variables[i])] = agc(trainable_variables[i], grad) 
                
                step = tf.cast(self.iterations + 1, trainable_variables[i].dtype)
                if i + self.rank < len(trainable_variables):
                    p = trainable_variables[i + self.rank]
                    
                    lr = tf.cast(learning_rate, p.dtype)
                    
                    if self.weight_decouple:
                        trainable_variables[i].assign(trainable_variables[i] * (1.0 - self.weight_decay * lr))
                    elif self.weight_decay > 0.0:
                        grad += trainable_variables[i] * self.weight_decay

                    if self.maximize:
                        gradient = -grads[self._get_variable_index(trainable_variables[i])]

                    if self.weight_decouple:
                        trainable_variables[i].assign(trainable_variables[i] * (1.0 - self.weight_decay * lr))
                    elif self.weight_decay > 0.0:
                        gradient += trainable_variables[i] * self.weight_decay
                    
                    if not self.pnm:
                        buf = self.momentum_buffer[self._get_variable_index(trainable_variables[i])]
                        buf.assign(buf + (1.0 - self.momentum) * (gradient - buf))
                    else:
                        noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                        def true_fn():
                            return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                        def false_fn():
                            return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                        pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                        pos_momentum.assign(pos_momentum * self.beta1 ** 2 + grad * (1.0 - self.beta1 ** 2))
                        buf = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)

                    update = gradient + self.momentum * (buf - gradient) if self.nesterov else buf
                    if len(update.shape) > 2:
                        update = tf.reshape(update, (len(update), -1))

                    update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)

                    if self.cautions:
                        mask = tf.cast(update * grad > 0, grad.dtype)
                        mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                        update *= mask
                    
                    if self.trust_ratio:
                        # Layer-wise LR adaptation
                        if self.sn:
                            size = tf.size(p)
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

                    lr = self.get_adjusted_lr(lr, p.shape, use_adjusted_lr=self.use_adjusted_lr)
                    lr = tf.cast(lr, p.dtype)

                    trainable_variables[i].assign_add(tf.reshape(update, (p.shape)) * -lr)
                    
                    if self.lookahead:
                        def true_fn():
                            slow_p = self.slow_momentum[self._get_variable_index(p)]
                            slow_p.assign(slow_p + self.lookahead_blending_alpha * (trainable_variables[i] - slow_p))
                            trainable_variables[i].assign(slow_p)
                        
                        def false_fn():
                            pass
                    
                        tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)

                self.distributed_step(self.padded_params)
        else:
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
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'DistributedMuon_e does not support sparse gradients')
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad)
                lr = tf.cast(self.adamw_lr, p.dtype)
                step = tf.cast(self.iterations + 1, p.dtype)
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
                step_size = lr * bias_correction2_sq / bias_correction1
                if self.DAdapt:
                    d_lr = self.d0 * self.adamw_lr
                    d_lr = d_lr * bias_correction2_sq / bias_correction1
                    d_lr = tf.cast(d_lr, p.dtype)
                    s = self.s[self._get_variable_index(p)]
                
                beta1, beta2, beta3 = self.adamw_betas
                
                if self.aem:
                    beta1 = tf.cast(beta1, p.dtype)
                    beta3 = tf.cast(beta3, p.dtype)
                    
                    alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                    beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
                    
                    exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
            
                    clip = tf.pow(step, 0.25)
                
                grad = grads[self._get_variable_index(p)]
                
                size = tf.size(grad)
                
                if self.sn:
                    reshaped_grad = tf.reshape(grad, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                else:
                    second_moment_update = tf.pow(grad, 2)

                if not self.pnm:
                    exp_avg = self.exp_avg[self._get_variable_index(p)]
                if self.sophia:
                    hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                else:
                    exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                                
                if not self.aem:
                    normed_grad = grad
                else:
                    normed_grad = tf.clip_by_value(
                        grad / tf.maximum(tf.sqrt(exp_avg_sq), self.adamw_eps if self.adamw_eps is not None else 1e-8),
                        clip_value_min=-clip,
                        clip_value_max= clip,
                    )
                
                if not self.pnm:
                    if self.DAdapt:
                        beta2_sq = math.sqrt(self.beta2)
                        exp_avg.assign(beta1 * exp_avg + (1.0 - beta1) * normed_grad * d_lr)
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        exp_avg.assign(beta1 * exp_avg + (1.0 - beta1) * normed_grad)
                else:
                    noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    if self.DAdapt:
                        beta2_sq = math.sqrt(beta2)
                        pos_momentum.assign(beta1**2 * pos_momentum + (1.0 - beta1**2) * normed_grad * d_lr)
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        pos_momentum.assign(beta1**2 * pos_momentum + (1.0 - beta1**2) * normed_grad)
                    exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                if self.sophia:
                    def true_fn2():
                        hessian_moment.assign(hessian_moment * beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - beta2))
                    def false_fn2():
                        pass
                    tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
                else:
                    exp_avg_sq.assign(beta2 * exp_avg_sq + (1.0 - beta2) * second_moment_update)
                
                if self.aem:
                    exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                
                if self.sophia:
                    de_nom = tf.maximum(hessian_moment, self.adamw_eps)
                else:
                    de_nom = tf.sqrt(exp_avg_sq) + self.adamw_eps
                
                if self.DAdapt:
                    flat_grad = tf.reshape(grad, [-1])
                    flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                    dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                    self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
                
                if not self.DAdapt:
                    if self.aem:
                        exp_avg += exp_avg_slow * alpha_t
                    if self.sn:
                        numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        normed_grad = tf.reshape(numerator / de_nom, p.shape)
                        if self.sophia:
                            update = tf.clip_by_value(normed_grad, clip_value_min=-p, clip_value_max=p)
                        if self.cautious:
                            mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                            numel = tf.cast(tf.size(mask), grad.dtype)
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
                    else:
                        normed_grad = exp_avg / de_nom
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        if self.cautious:
                            mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                            numel = tf.cast(tf.size(mask), grad.dtype)
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
                        p.assign_add(-step_size * update)
                    
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
                    beta1, beta2, beta3 = self.adamw_betas
                    step = self.iterations + 1
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2_sq = tf.sqrt(1 - beta2 ** step)
                    d_lr = self.d0 * self.adamw_lr
                    d_lr = d_lr * bias_correction2_sq / bias_correction1
                    
                    beta2_sq = math.sqrt(beta2)
                    
                    d = self.d0_
                    self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                    
                    if self.lr > 0.0:
                        d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                        d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                    
                    self.d0_.assign(d)
                    
                    for p in trainable_variables:
                        d_lr = tf.cast(d_lr, p.dtype)
                        
                        step = tf.cast(self.iterations + 1, p.dtype)
                        
                        if not self.pnm:
                            exp_avg = self.exp_avg[self._get_variable_index(p)]
                        else:
                            noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                            def true_fn():
                                return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                            def false_fn():
                                return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                            pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                            exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                        
                        if self.sophia:
                            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                        else:
                            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                        
                        if self.aem:
                            exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
                            exp_avg += exp_avg_slow * alpha_t
                        
                        if self.sophia:
                            de_nom = tf.maximum(hessian_moment, self.adamw_eps)
                        else:
                            de_nom = tf.sqrt(exp_avg_sq) + self.adamw_eps
                            
                        if self.sn:
                            numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                            normed_grad = tf.reshape(numerator / de_nom, p.shape)
                            if self.sophia:
                                update = tf.clip_by_value(normed_grad, clip_value_min=-p, clip_value_max=p)
                            if self.cautious:
                                mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                                numel = tf.cast(tf.size(mask), grad.dtype)
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
                        else:
                            normed_grad = exp_avg / de_nom
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                            if self.cautious:
                                mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                                numel = tf.cast(tf.size(mask), grad.dtype)
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
                            p.assign_add(-step_size * update)
                        
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
                "lr": self.lr,
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_betas": self.adamw_betas,
                "adamw_wd": self.adamw_wd,
                "adamw_eps": self.adamw_eps,
                "use_muon": self.use_muon,
                "cautious": self.cautious,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "aem": self.aem,
                "alpha": self.alpha,
                "t_alpha_beta3": self.t_alpha_beta3,
                "sophia": self.sophia,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "DAdapt": self.DAdapt,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
                "maximize": self.maximize,
                "world_size": self.world_size,
                "rank": self.rank,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class AdaMuon_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        beta3=0.9999,
        epsilon: float = 1e-8,
        weight_decay=1e-2,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_wd: float = 0.0,
        use_muon: bool = True,
        subset_size=-1,
        sn=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        agc=True,
        cautious=True,
        aem=False,
        alpha=5.0,
        t_alpha_beta3=None,
        sophia=True,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
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
        name="adamuon_e",
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
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_wd = adamw_wd
        self.use_muon = use_muon
        self.subset_size = subset_size
        self.sn = sn
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.aem = aem
        self.alpha = alpha
        self.t_alpha_beta3 = t_alpha_beta3
        self.sophia = sophia
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.m = []
        self.v = []
        self.exp_avg = []
        self.exp_avg_slow = []
        self.exp_avg_sq = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        self.hessian_moment = []
        self.hessian = []
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
            if self.use_muon:
                if not self.pnm:
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
                if self.trust_ratio and self.sn:
                    size = tf.size(var)
                    
                    def true_fn():
                        return self.subset_size
                    def false_fn():
                        return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                    self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                        size,
                        tf.cond(self.subset_size > 0, true_fn, false_fn)
                    ))
            else:
                if not self.pnm:
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
                if self.aem:
                    self.exp_avg_slow.append(self.add_variable_from_reference(
                        reference_variable=var, name="moment1_slow"
                                            ))
                if self.sophia:
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=var, name="hessian_moment"
                                            ))
                    self.hessian.append(self.add_variable_from_reference(
                                        reference_variable=var, name="hessian"
                                                            ))
                if self.DAdapt:
                    self.s.append(self.add_variable_from_reference(
                        reference_variable=var, name="s"
                                            ))
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
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
    
    def compute_hutchinson_hessian(
        self,
        params,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')
    
        params = [p for p in params if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [grads[self._get_variable_index(p)] for p in params if not tf.keras.backend.is_sparse(grads[self._get_variable_index(p)])]
    
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
    
    def apply_gradients(self, grads_and_vars, tape=None):
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
        if self.use_muon:
            for i, p in enumerate(trainable_variables):
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        ' AdaMuon_e does not support sparse gradients')
                
                lr = tf.cast(learning_rate, p.dtype)
                    
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.weight_decay * lr))
                elif self.weight_decay > 0.0:
                    grad += p * self.weight_decay
                
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad)
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                if not self.pnm:
                    buf = self.momentum_buffer[self._get_variable_index(p)]
                    buf.assign(buf + 1.0 - self.beta1 * (grad - buf))
                else:
                    noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    pos_momentum.assign(pos_momentum + (1.0 - self.beta1**2) * (grad - pos_momentum))
                    buf = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                
                update = grad + self.beta1 * (buf - grad) if self.nesterov else buf
                
                if len(update.shape) > 2:
                    update = tf.reshape(update, (len(update), -1))
                
                update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
                
                v = self.v[self._get_variable_index(p)]
                v.assign(v * self.beta2 + grad * grad * (1.0 - self.beta2))
                
                step = tf.cast(self.iterations + 1, p.dtype)
                bias_correction2 = 1 - self.beta2 ** step
                
                update = grad / tf.sqrt(v / bias_correction2) + self.epsilon
                update = tf.reshape(update, p.shape)
                
                update = update * 0.2 * math.sqrt(np.prod(p.shape.as_list())) / tf.norm(update) + self.epsilon
    
                lr = self.get_adjusted_lr(lr, p.shape, self.use_adjusted_lr)
    
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                    numel = tf.cast(tf.size(mask), grad.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    if self.sn:
                        size = tf.size(p)
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
                
                p.assign_add(-lr * update)
                
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
        else:
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
                
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        ' AdaMuon_e does not support sparse gradients')
                
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad) 
                    
                lr = tf.cast(self.adamw_lr, p.dtype)
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2 = 1 - self.beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
                step_size = lr / scale
                if self.DAdapt:
                    d_lr = self.d0 * self.adamw_lr
                    d_lr = d_lr / scale
                    d_lr = tf.cast(d_lr, p.dtype)
                    s = self.s[self._get_variable_index(p)]
                
                if self.aem:
                    beta1 = tf.cast(self.beta1, p.dtype)
                    beta3 = tf.cast(self.beta3, p.dtype)
    
                    alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                    beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
    
                    exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
    
                    clip = tf.pow(step, 0.25)
                
                size = tf.size(grad)
                
                if not self.pnm:
                    buf1 = self.exp_avg[self._get_variable_index(p)]
                if self.sophia:
                    hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                else:
                    buf2 = self.exp_avg_sq[self._get_variable_index(p)]
                if not self.aem:
                    normed_grad = grad
                else:
                    normed_grad = tf.clip_by_value(
                        grad / tf.maximum(tf.sqrt(buf2), self.adamw_eps if self.adamw_eps is not None else 1e-8),
                        clip_value_min=-clip,
                        clip_value_max= clip,
                    )
                if self.sn:
                    reshaped_grad = tf.reshape(grad, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                else:
                    second_moment_update = tf.pow(grad, 2)
                if not self.pnm:
                    if self.DAdapt:
                        beta2_sq = math.sqrt(self.beta2)
                        buf1.assign(buf1 + (1.0 - self.beta1) * (normed_grad * d_lr - buf1))
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        buf1.assign(buf1 + (1.0 - self.beta1) * (normed_grad - buf1))
                else:
                    noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    if self.DAdapt:
                        beta2_sq = math.sqrt(self.beta2)
                        pos_momentum.assign(pos_momentum * (1.0 - self.beta1**2) * (normed_grad * d_lr - pos_momentum))
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        pos_momentum.assign(pos_momentum * (1.0 - self.beta1**2) * (normed_grad - pos_momentum))
                    buf1 = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                if self.sophia:
                    def true_fn2():
                        hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))
                    def false_fn2():
                        pass
                    tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
                else:
                    buf2.assign(buf2 + (1.0 - self.beta2) * (second_moment_update - buf2))
                
                if self.aem:
                    exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                
                if self.sophia:
                    de_nom = tf.maximum(hessian_moment, self.epsilon)
                else:
                    de_nom = tf.sqrt(buf2) + self.epsilon
                
                if self.DAdapt:
                    flat_grad = tf.reshape(grad, [-1])
                    flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                    dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                    self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
    
                if not self.DAdapt:
                    if self.weight_decouple:
                        p.assign(p * (1.0 - self.weight_decay * lr))
                    elif self.weight_decay > 0.0:
                        grad += p * self.weight_decay
                    if self.aem:
                        buf1 += exp_avg_slow * alpha_t
                    if self.sn:
                        numerator = tf.reshape(buf1, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        normed_grad = tf.reshape(numerator / de_nom, p.shape)
                        update = normed_grad
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                    else:
                        update = buf1 / de_nom
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                    
                    if self.cautious:
                        mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                        numel = tf.cast(tf.size(mask), grad.dtype)
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
                
            if self.DAdapt:
                def update_fn():
                    step = self.iterations + 1
                    bias_correction1 = 1 - self.beta1 ** step
                    bias_correction2 = 1 - self.beta2 ** step
                    scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
                    d_lr = self.d0 * self.adamw_lr
                    
                    beta2_sq = math.sqrt(self.beta2)
                    
                    d = self.d0_
                    self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                    
                    if self.lr > 0.0:
                        d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                        d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                    
                    self.d0_.assign(d)
                    
                    for p in trainable_variables:
                        d_lr = tf.cast(d_lr, p.dtype)
                        
                        step = tf.cast(self.iterations + 1, p.dtype)
                        
                        grad = grads[self._get_variable_index(p)]
                        
                        if self.weight_decouple:
                            p.assign(p * (1.0 - self.adamw_wd * d_lr))
                        elif self.adamw_wd > 0.0:
                            grad += p * self.adamw_wd
                        
                        if not self.pnm:
                            buf1 = self.exp_avg[self._get_variable_index(p)]
                        else:
                            noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                            def true_fn():
                                return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                            def false_fn():
                                return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                            pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                            buf1 = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                        
                        if self.sophia:
                            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                        else:
                            buf2 = self.exp_avg_sq[self._get_variable_index(p)]
                        
                        if self.aem:
                            exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
                            buf1 += exp_avg_slow * alpha_t
                        
                        if self.sophia:
                            de_nom = tf.maximum(hessian_moment, self.epsilon)
                        else:
                            de_nom = tf.sqrt(buf2) + self.epsilon
                            
                        if self.sn:
                            numerator = tf.reshape(buf1, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                            normed_grad = tf.reshape(numerator / de_nom, p.shape)
                            update = normed_grad
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        else:
                            normed_grad = buf1 / de_nom
                            update = normed_grad
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        if self.cautious:
                            mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                            numel = tf.cast(tf.size(mask), grad.dtype)
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
                        
                        step_size = d_lr / scale
            
                        p.assign_add(update * -step_size)
                        
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
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_wd": self.adamw_wd,
                "use_muon": self.use_muon,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "cautious": self.cautious,
                "aem": self.aem,
                "alpha": self.alpha,
                "t_alpha_beta3": self.t_alpha_beta3,
                "sophia": self.sophia,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
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


class AdaGO_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=5e-2,
        epsilon=5e-4,
        weight_decay=0.0,
        momentum=0.95,
        weight_decouple=True,
        gamma=10.0,
        v=1e-6,
        nesterov=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_betas=(0.9,0.95,0.9999),
        adamw_wd=0.0,
        adamw_eps=1e-10,
        maximize=False,
        use_muon=True,
        subset_size=-1,
        sn=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        agc=True,
        cautious=True,
        aem=False,
        alpha=5.0,
        t_alpha_beta3=None,
        sophia=True,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
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
        name="adago_e",
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
        self.epsilon = epsilon
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.gamma = gamma
        self.v = v
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.use_adjusted_lr = use_adjusted_lr
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_wd = adamw_wd
        self.adamw_eps = adamw_eps
        self.maximize = maximize
        self.use_muon = use_muon
        self.subset_size = subset_size
        self.sn = sn
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.cautious = cautious
        self.aem = aem
        self.alpha = alpha
        self.t_alpha_beta3 = t_alpha_beta3
        self.sophia = sophia
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.v_ = []
        self.exp_avg = []
        self.exp_avg_slow = []
        self.exp_avg_sq = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        self.hessian_moment = []
        self.hessian = []
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
            if self.use_muon:
                if not self.pnm:
                    self.momentum_buffer.append(self.add_variable_from_reference(
                        reference_variable=var, name="momentum_buffer"
                                            ))
                self.v_.append(tf.Variable(self.v, dtype=var.dtype))
                self._track_variable(self.v_[-1])
                if self.trust_ratio and self.sn:
                    size = tf.size(var)
                    
                    def true_fn():
                        return self.subset_size
                    def false_fn():
                        return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                    self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                        size,
                        tf.cond(self.subset_size > 0, true_fn, false_fn)
                    ))
            else:
                if not self.pnm:
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
                if self.aem:
                    self.exp_avg_slow.append(self.add_variable_from_reference(
                        reference_variable=var, name="moment1_slow"
                                            ))
                if self.sophia:
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=var, name="hessian_moment"
                                            ))
                    self.hessian.append(self.add_variable_from_reference(
                                        reference_variable=var, name="hessian"
                                                            ))
                if self.DAdapt:
                    self.s.append(self.add_variable_from_reference(
                        reference_variable=var, name="s"
                                            ))
        self.adamw_lr = tf.Variable(self.adamw_lr)
        self._track_variable(self.adamw_lr)
    
    @staticmethod
    def get_adjusted_lr(lr, param_shape, use_adjusted_lr = False):
        r"""Get the adjust learning rate."""
        output_shape, *input_shape = param_shape
        input_shape = tf.reduce_prod(input_shape)

        ratio = (
            tf.pow(tf.maximum(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * tf.sqrt(tf.maximum(output_shape, input_shape))
        )

        return lr * ratio
    
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
    
    def compute_hutchinson_hessian(
        self,
        params,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params = [p for p in params if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [grads[self._get_variable_index(p)] for p in params if not tf.keras.backend.is_sparse(grads[self._get_variable_index(p)])]

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
    
    def apply_gradients(self, grads_and_vars, tape=None):
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
        if self.use_muon:
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'AdaGO_e does not support sparse gradients')
                
                lr = tf.cast(learning_rate, p.dtype)
                
                step = tf.cast(self.iterations + 1, p.dtype)
                
                beta1, beta2, _ = self.adamw_betas
                
                if self.maximize:
                    grad = -grad
                
                if self.weight_decouple:
                    p.assign(p * (1.0 - self.weight_decay * lr))
                elif self.weight_decay > 0.0:
                    grad += p * self.weight_decay
                    
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad) 
                
                buf = self.momentum_buffer[self._get_variable_index(p)]
                v = self.v_[self._get_variable_index(p)]
                buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)
                
                if not self.pnm:
                    buf = self.momentum_buffer[self._get_variable_index(p)]
                    buf.assign(buf * self.momentum + (1.0 - self.momentum) * grad)
                else:
                    noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    pos_momentum.assign(pos_momentum * beta1 ** 2 + grad * (1.0 - beta1 ** 2))
                    buf = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                
                v.assign_add(tf.minimum(tf.pow(tf.norm(grad, ord=2.0), 2), self.gamma ** 2))
                
                update = grad + self.momentum * (buf - grad) if self.nesterov else buf
                
                if len(update.shape) > 2:
                    update = tf.reshape(update, (len(update), -1))
                
                update = zero_power_via_newton_schulz_5(update, num_steps=self.ns_steps)
    
                lr = self.get_adjusted_lr(lr, p.shape, self.use_adjusted_lr)
                
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                    numel = tf.cast(tf.size(mask), grad.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    if self.sn:
                        size = tf.size(p)
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
    
                p.assign_add(tf.reshape(update, p.shape) * -tf.maximum(self.epsilon, lr * tf.minimum(tf.norm(grad, ord=2), self.gamma) / v))
                
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
        else:
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
                
            for p in trainable_variables:
                grad = grads[self._get_variable_index(p)]
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'AdaGO_e does not support sparse gradients')
                    
                if self.agc:
                    grads[self._get_variable_index(p)] = agc(p, grad) 
                    
                beta1, beta2, beta3 = self.adamw_betas
                    
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction2_sq = tf.sqrt(bias_correction2)
                
                lr = tf.cast(self.adamw_lr, p.dtype)
                step_size = lr / bias_correction1
                if self.DAdapt:
                    d_lr = self.d0 * self.adamw_lr
                    d_lr = d_lr / bias_correction1
                    d_lr = tf.cast(d_lr, p.dtype)
                    s = self.s[self._get_variable_index(p)]
                step = tf.cast(self.iterations + 1, p.dtype)
                
                if self.aem:
                    beta1 = tf.cast(beta1, p.dtype)
                    beta3 = tf.cast(beta3, p.dtype)
                    
                    alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                    beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
                    
                    exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
                    
                    clip = tf.pow(step, 0.25)
                
                size = tf.size(grad)
                
                if not self.pnm:
                    exp_avg = self.exp_avg[self._get_variable_index(p)]
                if self.sophia:
                    hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                else:
                    exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                if not self.aem:
                    normed_grad = grad
                else:
                    normed_grad = tf.clip_by_value(
                        grad / tf.maximum(tf.sqrt(exp_avg_sq), self.adamw_eps if self.adamw_eps is not None else 1e-8),
                        clip_value_min=-clip,
                        clip_value_max= clip,
                    )
                if self.sn:
                    reshaped_grad = tf.reshape(grad, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                else:
                    second_moment_update = tf.pow(grad, 2)
                if not self.pnm:
                    if self.DAdapt:
                        beta2_sq = math.sqrt(beta2)
                        exp_avg.assign(exp_avg * beta1 + (1.0 - beta1) * normed_grad * d_lr)
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        exp_avg.assign(exp_avg * beta1 + (1.0 - beta1) * normed_grad * d_lr)
                else:
                    noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                    def true_fn():
                        return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                    pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                    if self.DAdapt:
                        beta2_sq = math.sqrt(beta2)
                        pos_momentum * beta1 + (1.0 - beta1) * normed_grad * d_lr
                        pos_momentum.assign(pos_momentum * (1.0 - beta1**2) * (normed_grad * d_lr - pos_momentum))
                        s.assign(s * beta2_sq + normed_grad * d_lr * (1.0 - beta2_sq))
                        self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                    else:
                        pos_momentum.assign(pos_momentum * (1.0 - beta1**2) * (normed_grad - pos_momentum))
                    exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                if self.sophia:
                    def true_fn2():
                        hessian_moment.assign(hessian_moment * beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - beta2))
                    def false_fn2():
                        pass
                    tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
                else:
                    exp_avg_sq.assign(exp_avg_sq * beta2 + (1.0 - beta2) * second_moment_update)
                
                if self.aem:
                    exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                
                if self.sophia:
                    de_nom = tf.maximum(hessian_moment, self.adamw_eps)
                else:
                    de_nom = (tf.sqrt(exp_avg_sq) + self.adamw_eps) / bias_correction2_sq
                    
                if self.DAdapt:
                    flat_grad = tf.reshape(grad, [-1])
                    flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                    dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                    self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
                
                if not self.DAdapt:
                    if self.weight_decouple:
                        p.assign(p * (1.0 - self.adamw_wd * lr))
                    elif self.adamw_wd > 0.0:
                        grads[self._get_variable_index(p)] += p * self.adamw_wd
                    if self.aem:
                        exp_avg += exp_avg_slow * alpha_t
                    if self.sn:
                        numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        normed_grad = tf.reshape(numerator / de_nom, p.shape)
                        update = normed_grad
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                    else:
                        normed_grad = exp_avg / de_nom
                        update = normed_grad
                        if self.sophia:
                            update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                    if self.cautious:
                        mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                        numel = tf.cast(tf.size(mask), grad.dtype)
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
                        
                    p.assign_add(-step_size * update)
                    
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
                    beta1, beta2, _ = self.adamw_betas
                    step = self.iterations + 1
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    bias_correction2_sq = tf.sqrt(bias_correction2)
                    d_lr = self.d0 * self.adamw_lr
                    
                    beta2_sq = math.sqrt(beta2)
                    
                    d = self.d0_
                    self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                    
                    if self.lr > 0.0:
                        d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                        d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                    
                    self.d0_.assign(d)
                    
                    for p in trainable_variables:
                        d_lr = tf.cast(d_lr, p.dtype)
                        
                        step = tf.cast(self.iterations + 1, p.dtype)
                        
                        grad = grads[self._get_variable_index(p)]
                        
                        if self.weight_decouple:
                            p.assign(p * (1.0 - self.adamw_wd * d_lr))
                        elif self.adamw_wd > 0.0:
                            grad += p * self.adamw_wd
                        
                        if not self.pnm:
                            exp_avg = self.exp_avg[self._get_variable_index(p)]
                        else:
                            noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                            def true_fn():
                                return self.pos_momentum[self._get_variable_index(p)], self.neg_momentum[self._get_variable_index(p)]
                            def false_fn():
                                return self.neg_momentum[self._get_variable_index(p)], self.pos_momentum[self._get_variable_index(p)]
                            pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                            exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
                        
                        if self.sophia:
                            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
                        else:
                            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                        
                        if self.aem:
                            exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
                            exp_avg += exp_avg_slow * alpha_t
                        
                        if self.sophia:
                            de_nom = tf.maximum(hessian_moment, self.adamw_eps)
                        else:
                            de_nom = (tf.sqrt(exp_avg_sq) + self.adamw_eps) / bias_correction2_sq
                            
                        if self.sn:
                            numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                            normed_grad = tf.reshape(numerator / de_nom, p.shape)
                            update = normed_grad
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        else:
                            normed_grad = exp_avg / de_nom
                            update = normed_grad
                            if self.sophia:
                                update = tf.clip_by_value(update, clip_value_min=-p, clip_value_max=p)
                        if self.cautious:
                            mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                            numel = tf.cast(tf.size(mask), grad.dtype)
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
                        
                        step_size = d_lr / bias_correction1
            
                        p.assign_add(update * -step_size)
                        
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
                "lr": self.lr,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "gamma": self.gamma,
                "v": self.v,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "use_adjusted_lr": self.use_adjusted_lr,
                "adamw_lr": self.adamw_lr,
                "adamw_betas": self.adamw_betas,
                "adamw_wd": self.adamw_wd,
                "adamw_eps": self.adamw_eps,
                "maximize": self.maximize,
                "use_muon": self.use_muon,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "cautious": self.cautious,
                "aem": self.aem,
                "alpha": self.alpha,
                "t_alpha_beta3": self.t_alpha_beta3,
                "sophia": self.sophia,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
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
