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
            self.use_muon[self._get_variable_index(p)] = len(p.shape) >= 2

        for p in adamw_params:
            self.use_muon[self._get_variable_index(p)] = False
    
    def reset(self):
        self.step = 0
        iterations = tf.Variable(
                0,
                name="iteration",
                dtype=tf.int64,
                trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
        self._track_variable(iterations)
        self._iterations = iterations
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
        self.step = 0
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
        self.step += 1
        
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
        
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step
        scale = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
        step_size = lr / scale
        
        for p in params:
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
                "step": self.iterations.numpy(),
            }
        )
        return config
    
    def _update_step(self):
        if hasattr(self, 'step'):
            if type(self.step) == list:
                self.step = [self.iterations.numpy() for _ in range(len(self.step))]
            else:
                self.step = self.iterations.numpy()
	
    def _apply_weight_decay(self, variables):
        pass