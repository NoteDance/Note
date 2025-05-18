""" Fira
https://arxiv.org/abs/2410.01623

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from Note.nn.optimizer.galore_projector import GaLoreProjector


class Fira(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        epsilon=1e-6,
        weight_decay=0.0,
        maximize=False,
        rank=None,
        update_proj_gap=None,
        scale=None,
        projection_type=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="fira",
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
        self.betas = betas
        self.epsilon = epsilon
        self.maximize = maximize
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_inf"
                                                    )
            self.scaling_grad[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="scaling_grad"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.scaling_grad = []
        self.projector = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            self.scaling_grad.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="scaling_grad"
                )
            )
            self.projector.append(None)

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Fira does not support sparse gradient.')
        
        if variable.dtype.is_complex:
            raise RuntimeError(
                'Fira does not support complex parameter.')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2 = self.betas
        
        bias_correction1 = 1 - beta1 ** step
        bias_correction2_sq = tf.sqrt(1 - beta2 ** step)
        
        step_size = lr * bias_correction2_sq / bias_correction1
        
        if self.maximize:
            gradient = -gradient
        
        if self.rank is not None and len(variable.shape) == 2:
            if self.projector[self._get_variable_index(variable)] is None:
                self.projector[self._get_variable_index(variable)] = GaLoreProjector(
                    rank=self.rank,
                    update_proj_gap=self.update_proj_gap,
                    scale=self.scale,
                    projection_type=self.projection_type,
                )
        
            gradient = self.projector[self._get_variable_index(variable)].project(gradient, step)
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
        exp_avg_sq.assign(exp_avg_sq * beta2 + gradient * gradient * (1.0 - beta2))

        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon

        norm_grad = exp_avg / de_nom
        
        if self.rank is not None and len(variable.shape) == 2:
            sub_grad = self.projector[self._get_variable_index(variable)].project_back(gradient)

            norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1

            scaling_factor = tf.norm(norm_grad, axis=norm_dim) / (tf.norm(gradient, axis=norm_dim) + 1e-8)
            if norm_dim == 1:
                scaling_factor = tf.expand_dims(scaling_factor, axis=1)

            scaling_grad = (gradient - sub_grad) * scaling_factor
            
            def true_fn():
                self.scaling_grad[self._get_variable_index(variable)].assign(tf.norm(scaling_grad))
                return scaling_grad
            
            def false_fn(scaling_grad = scaling_grad):
                scaling_grad_norm = tf.norm(scaling_grad)

                limiter = tf.maximum(scaling_grad_norm / (self.scaling_grad[self._get_variable_index(variable)] + 1e-8), 1.01) / 1.01
                scaling_grad /= limiter

                self.scaling_grad[self._get_variable_index(variable)].assign(scaling_grad_norm / limiter)
                return scaling_grad

            scaling_grad = tf.cond(step == 1, true_fn, false_fn)
            
            norm_grad = self.projector[self._get_variable_index(variable)].project_back(norm_grad) + scaling_grad

        variable.assign_add(norm_grad * -step_size)
        
        variable.assign(variable * (1.0 - self.weight_decay * lr))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "maximize": self.maximize,
                "rank": self.rank,
                "update_proj_gap": self.update_proj_gap,
                "scale": self.scale,
                "projection_type": self.projection_type,
                "projector": self.projector,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass