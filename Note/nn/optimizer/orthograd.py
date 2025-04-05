""" OrthoGrad
https://arxiv.org/abs/2501.04697

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class OrthoGrad(optimizer.Optimizer):
    def __init__(
        self,
        base_optimizer=None,
        name="orthograd",
    ):
        super().__init__(
            learning_rate=1.,
            name=name,
        )
        self.base_optimizer = base_optimizer
    
    def reset(self):
        pass

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
    
    @staticmethod
    def _orthogonalize_gradients(self, params, grads):
        """
        Projects the gradient g to be orthogonal to the current weights w.

        g_orth = g - ( (w·g)/(w·w + eps) ) * w

        And then re-scales g_orth to have the same norm as g.
        """
        for p, g in zip(params, grads):
            w = tf.reshape(p, [-1])
            g = tf.reshape(g, [-1])

            w_norm_sq = tf.tensordot(w, w, axes=1) + 1e-30
            proj = tf.tensordot(w, g, axes=1) / w_norm_sq
            g_orth = g - proj * w

            g_norm = tf.norm(g, ord=2)
            g_orth_norm = tf.norm(g_orth, ord=2) + 1e-30
            g_orth_scaled = g_orth * (g_norm / g_orth_norm)

            grads[self._get_variable_index(p)] = tf.reshape(g_orth_scaled, g.shape)
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)
    
    def apply_gradients(self, grads_and_vars, tape=None):
        self.tape = tape
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations

    def update_step(self, grads, trainable_variables, learning_rate):
        self._orthogonalize_gradients(trainable_variables, grads)
        if self.tape is None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables))
        else:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_optimizer": self.base_optimizer,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass