""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610

Hacked together by / Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class Lookahead(optimizer.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6, name="lookahead"):
        super().__init__(learning_rate=1.,name=name)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self._base_optimizer = base_optimizer
        self.lookahead_alpha = alpha
        self.lookahead_k = k
        self.lookahead_step = 0
    
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._base_optimizer.lookahead_slow_buff = []
        for var in var_list:
            self._base_optimizer.lookahead_slow_buff.append(None)

    def update_slow(self, trainable_variables):
        for fast_p in trainable_variables:
            self._base_optimizer.lookahead_slow_buff[self._get_variable_index(fast_p)] = tf.zeros_like(fast_p)
            self._base_optimizer.lookahead_slow_buff[self._get_variable_index(fast_p)] = fast_p
            slow = self._base_optimizer.lookahead_slow_buff[self._get_variable_index(fast_p)]
            self._base_optimizer.lookahead_slow_buff[self._get_variable_index(fast_p)] += (fast_p - slow) * self.lookahead_alpha
            fast_p.assign(slow)

    def sync_lookahead(self):
        self.update_slow()
    
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
        if self.tape is None:
            self._base_optimizer.apply_gradients(zip(grads, trainable_variables))
        else:
            self._base_optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        self.lookahead_step += 1
        if self.lookahead_step % self.lookahead_k == 0:
            self.update_slow(trainable_variables)

    def state_dict(self):
        return tf.keras.optimizers.serialize(self._base_optimizer)

    def load_state_dict(self, state_dict):
        self._base_optimizer=tf.keras.optimizers.deserialize(state_dict)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lookahead_alpha": self.lookahead_alpha,
                "lookahead_k": self.lookahead_k,
                "lookahead_step": self.lookahead_step,
                "base_optimizer": tf.keras.optimizers.serialize(self._base_optimizer),
            }
        )
        return config