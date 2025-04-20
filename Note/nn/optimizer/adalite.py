""" Adalite
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class Adalite(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-2,
        weight_decouple=False,
        fixed_decay=False,
        g_norm_min=1e-10,
        ratio_min=1e-4,
        tau=1.0,
        eps1=1e-6,
        eps2=1e-10,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adalite",
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
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.g_norm_min = g_norm_min
        self.ratio_min = ratio_min
        self.tau = tau
        self.eps1 = eps1
        self.eps2 = eps2
    
    def reset(self):
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
            if len(var.shape) < 2:
                self.m_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="m_avg"
                                                        )
                self.v_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="v_avg"
                                                        )
            else:
                self.v_avg_0[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(tf.reduce_mean(var, axis=1)), name="v_avg_0"
                                                        )
                self.v_avg_1[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(tf.reduce_mean(var, axis=0)), name="v_avg_1"
                                                        )

                self.m_avg_c[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(tf.reduce_mean(var, axis=1)[:, None]), name="m_avg_c"
                                                        )
                self.m_avg_r[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(tf.reduce_mean(var, axis=0)[None, :]), name="m_avg_r"
                                                        )
                self.m_avg_u[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(
                                                            tf.expand_dims(tf.expand_dims(tf.reduce_mean(var), 0), 0)), 
                                                            name="m_avg_u"
                                                        )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.m_avg = []
        self.v_avg = []
        self.v_avg_0 = []
        self.v_avg_1 = []
        self.m_avg_c = []
        self.m_avg_r = []
        self.m_avg_u = []
        self.step = []
        for var in var_list:
            self.m_avg.append(tf.Variable(0))
            self.v_avg.append(tf.Variable(0))
            self.v_avg_0.append(tf.Variable(0))
            self.v_avg_1.append(tf.Variable(0))
            self.m_avg_c.append(tf.Variable(0))
            self.m_avg_r.append(tf.Variable(0))
            self.m_avg_u.append(tf.Variable(0))
            if len(var.shape) < 2:
                self.m_avg[-1] = self.add_variable_from_reference(
                                    reference_variable=var, name="m_avg"
                                                        )
                self.v_avg[-1] = self.add_variable_from_reference(
                                    reference_variable=var, name="v_avg"
                                                        )
                self._track_variable(self.v_avg_0[-1])
                self._track_variable(self.v_avg_1[-1])
                self._track_variable(self.m_avg_c[-1])
                self._track_variable(self.m_avg_r[-1])
                self._track_variable(self.m_avg_u[-1])
            else:
                self.v_avg_0[-1] = self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.reduce_mean(var, axis=1)), name="v_avg_0"
                                                        )
                self.v_avg_1[-1] = self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.reduce_mean(var, axis=0)), name="v_avg_1"
                                                        )

                self.m_avg_c[-1] = self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.reduce_mean(var, axis=1)[:, None]), name="m_avg_c"
                                                        )
                self.m_avg_r[-1] = self.add_variable_from_reference(
                                    reference_variable=tf.Variable(tf.reduce_mean(var, axis=0)[None, :]), name="m_avg_r"
                                                        )
                self.m_avg_u[-1] = self.add_variable_from_reference(
                                    reference_variable=tf.Variable(
                                    tf.expand_dims(tf.expand_dims(tf.reduce_mean(var), 0), 0)), 
                                    name="m_avg_u"
                                                        )
                self._track_variable(self.m_avg[-1])
                self._track_variable(self.v_avg[-1])
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Adalite does not support sparse gradients')
        
        if sum(gradient.shape) > 1:
            p_norm = tf.norm(variable)
            grad_norm = tf.norm(gradient)
            grad_norm = tf.maximum(grad_norm, self.g_norm_min)
            trust_ratio = p_norm / grad_norm
            trust_ratio = tf.maximum(trust_ratio, self.ratio_min)
            gradient = gradient * trust_ratio

        if len(gradient.shape) < 2:
            m = self.m_avg[self._get_variable_index(variable)]
            v = self.v_avg[self._get_variable_index(variable)]
        else:
            r = self.v_avg_0[self._get_variable_index(variable)][:, None]
            c = self.v_avg_1[self._get_variable_index(variable)][None, :]
            r_sum = tf.reduce_sum(r)
            r_sum = tf.maximum(r_sum, self.eps2)
            v = (r * c) / r_sum
            m = tf.matmul(tf.matmul(self.m_avg_c[self._get_variable_index(variable)], 
                                    self.m_avg_u[self._get_variable_index(variable)]), 
                          self.m_avg_r[self._get_variable_index(variable)])

        m = m + (gradient - m) * (1.0 - self.beta1)
        v = v + (((gradient - m) ** 2) - v) * (1.0 - self.beta2)

        v_avg = v / (1.0 - self.beta2 ** self.step[self._get_variable_index(variable)])

        if len(gradient.shape) == 2:
            imp_c = tf.nn.softmax(tf.reduce_mean(v, axis=1), axis=0)[:, None]
            imp_r = tf.nn.softmax(tf.reduce_mean(v, axis=0), axis=0)[None, :]
            m = m + (gradient - m) * (1.0 - imp_c * imp_r)

        u = m + (gradient - m) * (1.0 - self.beta1)

        if len(gradient.shape) < 2:
            self.m_avg[self._get_variable_index(variable)] = m
            self.v_avg[self._get_variable_index(variable)] = v
        else:
            self.v_avg_0[self._get_variable_index(variable)] = tf.reduce_sum(v, axis=1)
            v_sum = tf.reduce_sum(v)
            v_sum = tf.maximum(v_sum, self.eps2)
            self.v_avg_1[self._get_variable_index(variable)] = tf.reduce_sum(v, axis=0) / v_sum
    
            imp_c = tf.nn.softmax(tf.reduce_mean(v, axis=1) / self.tau, axis=-1)[:, None]
            imp_r = tf.nn.softmax(tf.reduce_mean(v, axis=0) / self.tau, axis=-1)[None, :]
    
            c = tf.reduce_sum(m * imp_r, axis=1, keepdims=True)
            r = tf.reduce_sum(m * imp_c, axis=0, keepdims=True)
    
            s_num = tf.matmul(tf.matmul(tf.transpose(c), m), tf.transpose(r))
            s_den = tf.matmul(tf.matmul(tf.transpose(c), c), tf.matmul(r, tf.transpose(r)))
            s_den = tf.maximum(s_den, self.eps2)
            s = s_num / s_den
    
            self.m_avg_c[self._get_variable_index(variable)] = c
            self.m_avg_r[self._get_variable_index(variable)] = r
            self.m_avg_u[self._get_variable_index(variable)] = s
        
        u = u / tf.sqrt(v_avg + self.eps1)
        
        u = tf.reshape(u, variable.shape)
        u = u + self.weight_decay * variable
        
        variable.assign_add(-lr * u)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "g_norm_min": self.g_norm_min,
                "ratio_min": self.ratio_min,
                "tau": self.tau,
                "eps1": self.eps1,
                "eps2": self.eps2,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass