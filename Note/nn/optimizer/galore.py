""" GaLore
https://arxiv.org/abs/2403.03507

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from optimizers.galore_projector import GaLoreProjector
import math


class GaLore(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        weight_decay=0.0,
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
        name="galore",
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
    
    def reset(self):
        self.step = 0
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.projector = []
        self.step = 0
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.projector.append(None)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
                
        self.step += 1
        
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2_sq = math.sqrt(1 - self.beta2 ** self.step)
        
        step_size = lr * bias_correction2_sq / bias_correction1
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'GaLore does not support sparse gradients')
        
        if self.rank is not None and len(variable.shape) > 1:
            if self.projector[self._get_variable_index(variable)] is None:
                self.projector[self._get_variable_index(variable)] = GaLoreProjector(
                    rank=self.rank,
                    update_proj_gap=self.update_proj_gap,
                    scale=self.scale,
                    projection_type=self.projection_type,
                )

            grad = self.projector[self._get_variable_index(variable)].project(gradient, self.step)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + grad * grad * (1.0 - self.beta2))
        
        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        norm_grad = exp_avg / de_nom
    
        if self.rank is not None and len(variable.shape) > 1:
            norm_grad = self.projector[self._get_variable_index(variable)].project_back(norm_grad)
        
        variable.assign_add(norm_grad * -step_size)

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
                "step": self.step,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass