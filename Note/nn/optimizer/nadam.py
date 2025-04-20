import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class NAdam(optimizer.Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

    NOTE: This impl has been deprecated in favour of torch.optim.NAdam and remains as a reference

    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.

    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

        Originally taken from: https://github.com/pytorch/pytorch/pull/1408
        NOTE: Has potential issues but does work well on some problems.
    """

    def __init__(
        self,
        learning_rate=2e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0,
        schedule_decay=4e-3,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="nadam",
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
        self.schedule_decay = schedule_decay

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.m_schedule = []
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
            self.m_schedule.append(tf.Variable(1.))
            self._track_variable(self.m_schedule[-1])
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        # Warming momentum schedule
        m_schedule = self.m_schedule[self._get_variable_index(variable)]
        schedule_decay = self.schedule_decay
        exp_avg, exp_avg_sq = self.exp_avg[self._get_variable_index(variable)], self.exp_avg_sq[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2
        eps = self.epsilon
        self.step[self._get_variable_index(variable)] += 1
        t= self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - beta2 ** t
        
        if self.weight_decay != 0:
            gradient +=self.weight_decay * variable

        momentum_cache_t = beta1 * (1. - 0.5 * (0.96 ** (t * schedule_decay)))
        momentum_cache_t_1 = beta1 * (1. - 0.5 * (0.96 ** ((t + 1) * schedule_decay)))
        m_schedule_new = m_schedule * momentum_cache_t
        m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
        self.m_schedule[self._get_variable_index(variable)].assign(tf.cast(m_schedule_new, dtype=self.m_schedule[self._get_variable_index(variable)].dtype))
    
        # Decay the first and second moment running average coefficient
        exp_avg.assign(beta1 * exp_avg + (1. - beta1) * gradient)
        exp_avg_sq.assign(beta2 * exp_avg_sq + (1. - beta2) * tf.square(gradient))
    
        denom = (tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps
        variable.assign_add(-lr * (1. - momentum_cache_t) / (1. - m_schedule_new) * gradient / denom)
        variable.assign_add(-lr * momentum_cache_t_1 / (1. - m_schedule_next) * exp_avg / denom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "schedule_decay": self.schedule_decay,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass