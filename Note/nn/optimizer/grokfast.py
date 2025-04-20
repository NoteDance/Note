""" GrokFastAdamW
https://arxiv.org/abs/2405.20233

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from collections import deque
import math


class GrokFastAdamW(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        grokfast=True,
        grokfast_alpha=0.98,
        grokfast_lamb=2.0,
        grokfast_after_step=0,
        weight_decouple=True,
        fixed_decay=False,
        normalize_lr=True,
        filter='ma',
        filter_params=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="grokfastadamw",
        **kwargs,
    ):
        if grokfast and normalize_lr:
            learning_rate /= 1.0 + grokfast_lamb
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
        self.grokfast = grokfast
        self.grokfast_alpha = grokfast_alpha
        self.grokfast_lamb = grokfast_lamb
        self.grokfast_after_step = grokfast_after_step
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.normalize_lr = normalize_lr
        self.filter = filter
        self.filter_params = filter_params
        
        self.grads = None
    
    def gradfilter_ma(self, current_grads):
        if self.grads is None:
            for v in self._trainable_variables:
                self.grads = {self._get_variable_index(v): deque(maxlen=self.filter_params['window_size']) for v in self._trainable_variables}
        
        for v in self._trainable_variables:
            grad = current_grads[self._get_variable_index(v)]
            self.grads[self._get_variable_index(v)].append(grad)
            
            if (not self.filter_params['warmup']) or (len(self.grads[self._get_variable_index(v)]) == self.filter_params['window_size']):
                if self.filter_params['filter_type'] == 'mean':
                    avg = tf.add_n(list(self.grads[self._get_variable_index(v)])) / tf.cast(len(self.grads[self._get_variable_index(v)]), grad.dtype)
                elif self.filter_params['filter_type'] == 'sum':
                    avg = tf.add_n(list(self.grads[self._get_variable_index(v)]))
                else:
                    raise ValueError(f"not supported filter_type {self.filter_params['filter_type']}")
                    
                current_grads[self._get_variable_index(v)] += self.filter_params['lamb'] * avg

    def gradfilter_ema(self, current_grads):
        if self.grads is None:
            for v in self._trainable_variables:
                self.grads = {self._get_variable_index(v): current_grads[self._get_variable_index(v)]}
        
        for v in self._trainable_variables:
            grad = current_grads[self._get_variable_index(v)]
            self.grads[self._get_variable_index(v)] = self.grads[self._get_variable_index(v)] * self.filter_params['alpha'] + grad * (1.0 - self.filter_params['alpha'])
            current_grads[self._get_variable_index(v)] += self.filter_params['lamb'] * self.grads[self._get_variable_index(v)]
    
    def reset(self):
        self.grads = None
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
            self.exp_avg[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.grok_exp_avg = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.grok_exp_avg.append(None)
            self.step.append(0)
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        if self.filter == 'ma':
            self.gradfilter_ma(grads)
        elif self.filter == 'eam':
            self.gradfilter_ema(grads)
        for grad, var in zip(grads, trainable_variables):
            self.update_step(grad, var, learning_rate)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
                
        self.step[self._get_variable_index(variable)] += 1
        
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2_sq = math.sqrt(1 - self.beta2 ** self.step[self._get_variable_index(variable)])
        
        should_grokfast: bool = (
            self.grokfast and self.step[self._get_variable_index(variable)] > self.grokfast_after_step and self.grokfast_lamb > 0.0
        )
        
        if self.step[self._get_variable_index(variable)] == 1:
            self.grok_exp_avg[self._get_variable_index(variable)] = gradient
            self._track_variable(self.grok_exp_avg[self._get_variable_index(variable)])
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'GrokFastAdamW does not support sparse gradients')
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

        if should_grokfast:
            grok_exp_avg = self.grok_exp_avg[self._get_variable_index(variable)]
            grok_exp_avg = grok_exp_avg * self.grokfast_alpha + gradient * (1.0 - self.grokfast_alpha)
            
            gradient += grok_exp_avg * self.grokfast_lamb
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + gradient * gradient * (1.0 - self.beta2))

        de_nom = tf.maximum(tf.sqrt(exp_avg_sq) / bias_correction2_sq, self.epsilon)

        update = exp_avg / bias_correction1 / de_nom

        variable.assign_add(update * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "grokfast": self.grokfast,
                "grokfast_alpha": self.grokfast_alpha,
                "grokfast_lamb": self.grokfast_lamb,
                "grokfast_after_step": self.grokfast_after_step,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "normalize_lr": self.normalize_lr,
                "filter": self.filter,
                "filter_params": self.filter_params,
                "grads": self.grads,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass