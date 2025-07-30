""" SPlus
https://arxiv.org/abs/2506.07254

Copyright 2025 NoteDance
"""
from typing import Tuple

import tensorflow as tf
from keras.src.optimizers import optimizer


class SPlus(optimizer.Optimizer):
    r"""A Stable Whitening Optimizer for Efficient Neural Network Training.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ema_rate: float. exponential moving average decay rate.
    :param inverse_steps: int. the number of steps to perform inverse.
    :param nonstandard_constant: float. scale factor for learning rate in case of non-linear layer.
    :param max_dim: int. maximum number of dimensions to perform .
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params,
        learning_rate = 1e-1,
        betas = (0.9, 0.999),
        epsilon = 1e-30,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ema_rate: float = 0.999,
        inverse_steps: int = 100,
        nonstandard_constant: float = 1e-3,
        max_dim: int = 10000,
        maximize: bool = False,
        train_mode = True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name = "splus",
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
        self.params = params
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.ema_rate = ema_rate
        self.inverse_steps = inverse_steps
        self.nonstandard_constant = nonstandard_constant
        self.max_dim = max_dim
        self.maximize = maximize
        self.train_mode = train_mode

    def eval(self):
        if self.train_mode:
            for p in self.params:
                self.param_buffer[self._get_variable_index(p)].assign(p)
                p.assign(self.ema[self._get_variable_index(p)] * (1.0 / (1.0 - tf.pow(self.ema_rate, tf.cast(self.iterations + 1, p.dtype)))))
            self.train_mode = False

    def train(self):
        if not self.train_mode:
            for p in self.params:
                p.assign(self.param_buffer[self._get_variable_index(p)])
            self.train_mode = True
    
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum = []
        self.ema = []
        self.sides = []
        self.q_sides = []
        for var in var_list:
            self.momentum.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
            self.ema.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="ema"
                )
            )
            if len(var.shape) == 2:
                self.sides.append([
                    tf.Variable(tf.zeros((d, d), dtype=var.dtype)) if d < self.max_dim else None
                    for d in var.shape
                ]
                )
                self._track_variable(self.sides[-1])
                self.q_sides.append([
                    tf.Variable(tf.eye(d, dtype=var.dtype)) if d < self.max_dim else None
                    for d in var.shape
                ]
                )
                self._track_variable(self.q_sides[-1])

    @staticmethod
    def get_scaled_lr(shape: Tuple[int, int], lr: float, nonstandard_constant: float, max_dim: int = 10000) -> float:
        scale: float = (
            nonstandard_constant
            if len(shape) != 2 or shape[0] > max_dim or shape[1] > max_dim
            else 2.0 / (shape[0] + shape[1])
        )
        return lr * scale

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'SPlus does not support sparse gradients')
        
        lr = tf.cast(learning_rate, variable.dtype)
            
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2 = self.betas

        if self.maximize:
            gradient = -gradient

        scaled_lr = self.get_scaled_lr(
            variable.shape, lr, self.nonstandard_constant, self.max_dim
        )
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else scaled_lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

        m, ema = self.momentum[self._get_variable_index(variable)], self.ema[self._get_variable_index(variable)]
        m.assign(m + (1.0 - beta1) * (gradient - m))

        if len(variable.shape) == 2:
            sides, q_sides = self.sides[self._get_variable_index(variable)], self.q_sides[self._get_variable_index(variable)]

            m = tf.matmul(tf.transpose(q_sides[0]), m) if q_sides[0] is not None else m
            m = tf.matmul(m, q_sides[1]) if q_sides[1] is not None else m

            if sides[0] is not None:
                sides[0].assign(sides[0] + (1.0 - beta2) * (tf.matmul(gradient, gradient, transpose_b=True) - sides[0]))

            if sides[1] is not None:
                sides[1].assign(sides[1] + (1.0 - beta2) * (tf.matmul(gradient, gradient, transpose_a=True) - sides[1]))

            update = tf.sign(m)

            if q_sides[0] is not None:
                update = tf.matmul(q_sides[0], update)

            if q_sides[1] is not None:
                update = tf.matmul(update, tf.transpose(q_sides[1]))
            
            def true_fn():
                if sides[0] is not None:
                    _, eig_vecs = tf.linalg.eigh(
                        tf.cast(sides[0], tf.float32) + tf.eye(sides[0].shape[0]) * self.epsilon
                    )
                    self.q_sides[self._get_variable_index(variable)][0].assign(tf.cast(eig_vecs, sides[0].dtype))
                if sides[1] is not None:
                    _, eig_vecs = tf.linalg.eigh(
                        tf.cast(sides[1], tf.float32) + tf.eye(sides[1].shape[0]) * self.epsilon
                    )
                    self.q_sides[self._get_variable_index(variable)][1].assign(tf.cast(eig_vecs, sides[1].dtype))
            
            def false_fn():
                pass
            
            tf.cond(tf.logical_or(step == 1, step % self.inverse_steps == 0), true_fn, false_fn)
        else:
            update = tf.sign(m)

        variable.assign_add(update * -scaled_lr)

        ema.assign(ema + (1.0 - self.ema_rate) * (variable - ema))
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "ema_rate": self.ema_rate,
                "inverse_steps": self.inverse_steps,
                "nonstandard_constant": self.nonstandard_constant,
                "max_dim": self.max_dim,
                "maximize": self.maximize,
                "train_mode": self.train_mode,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass