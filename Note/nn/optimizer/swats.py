""" SWATS
Implements SWATS Optimizer Algorithm.
It has been proposed in `Improving Generalization Performance by
Switching from Adam to SGD`__.
https://arxiv.org/pdf/1712.07628.pdf

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class SWATS(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-3,
        weight_decay=0,
        amsgrad=False,
        nesterov=False,
        phase="ADAM",
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="swats",
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
        self.amsgrad = amsgrad
        self.nesterov = nesterov
        self.phase = phase
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.amsgrad = False
        self.nesterov = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg2 = []
        if self.amsgrad:
            self.max_exp_avg_sq = []
        self.momentum_buffer = []
        self.step = []
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
            self.exp_avg2.append(
                self.add_variable_from_reference(
                    reference_variable=tf.Variable(tf.zeros([1], dtype=var.dtype)), name="exp_avg2"
                )
            )
            if self.amsgrad:
                self.max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    )
                )
            self.momentum_buffer.append(None) 
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        exp_avg, exp_avg2, exp_avg_sq = (
            self.exp_avg[self._get_variable_index(variable)],
            self.exp_avg2[self._get_variable_index(variable)],
            self.exp_avg_sq[self._get_variable_index(variable)],
        )
        
        if self.amsgrad:
            max_exp_avg_sq = self.max_exp_avg_sq[self._get_variable_index(variable)]
        
        self.step[self._get_variable_index(variable)] += 1

        if self.weight_decay != 0:
            gradient += variable * self.weight_decay

        # if its SGD phase, take an SGD update and continue
        if tf.get_static_value(self.phase) == b"SGD":
            if self.momentum_buffer[self._get_variable_index(variable)] == None:
                buf = self.momentum_buffer[self._get_variable_index(variable)] = gradient
            else:
                buf = self.momentum_buffer[self._get_variable_index(variable)]
                self.momentum_buffer[self._get_variable_index(variable)] = buf * self.beta1 + gradient
                gradient = buf

            gradient = gradient * (1 - self.beta1)
            if self.nesterov:
                gradient += buf * self.beta1

            variable.assign_add(gradient * -lr)
            return
        
        # decay the first and second moment running average coefficient
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + (1 - self.beta2) * tf.square(gradient))
        if self.amsgrad:
            # maintains the maximum of all 2nd
            # moment running avg. till now
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
            # use the max. for normalizing running avg. of gradient
            denom = tf.sqrt(max_exp_avg_sq) + self.epsilon
        else:
            denom = tf.sqrt(exp_avg_sq) + self.epsilon

        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
        step_size = (
            lr * (bias_correction2**0.5) / bias_correction1
        )

        p = -step_size * (exp_avg / denom)
        variable.assign_add(p)

        p_view = tf.reshape(p, [-1])
        gradient_view = tf.reshape(gradient, [-1])
        pg = tf.tensordot(p_view, gradient_view, axes=1)
        
        def true_fn():
            # the non-orthognal scaling estimate
            scaling = tf.tensordot(p_view, p_view, axes=1) / (-pg)
            exp_avg2.assign(self.beta2 * exp_avg2 + (1 - self.beta2) * scaling)
        
            # bias corrected exponential average
            corrected_exp_avg = exp_avg2 / bias_correction2
                
            # checking criteria of switching to SGD training
            def true_fn():
                self.phase = "SGD"
            
            def false_fn():
                pass
            
            tf.cond((
                    tf.logical_and(tf.logical_and(tf.convert_to_tensor(self.step[self._get_variable_index(variable)]) > 1
                    ,tf.experimental.numpy.allclose(corrected_exp_avg, scaling, rtol=1e-6, atol=1e-8))
                    ,corrected_exp_avg > 0)
                    ), true_fn, false_fn)
            
        def false_fn():
            pass
            
        tf.cond(pg != 0, true_fn, false_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "nesterov": self.nesterov,
                "phase": self.phase,
                "momentum_buffer": self.momentum_buffer,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass