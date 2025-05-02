""" TRAC
https://arxiv.org/abs/2405.16642

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import numpy as np


def polyval(x, coef):
    result = tf.ones_like(x) * coef[0]
    for c in coef[1:]:
        result = result * x + c
    return result

class ERF1994:
    def __init__(self, num_coefs: int = 128):
        super().__init__()
        self.n: int = num_coefs
        self.m = 2 * self.n
        self.m2 = 2 * self.m
        self.k = np.linspace(-self.m + 1, self.m - 1, self.m2 - 1, dtype=np.float32)
        self.l = np.sqrt(self.n / np.sqrt(2.0))
        self.theta = self.k * np.pi / self.m
        self.t = self.l * np.tan(self.theta / 2.0)
        self.f = np.exp(-self.t**2) * (self.l**2 + self.t**2)
        self.a = np.fft.fftshift(np.fft.fft(self.f)).real / self.m2
        self.a = np.flipud(self.a[1 : self.n + 1])
        self.a = tf.constant(self.a, dtype=tf.complex64)
        self.l = tf.constant(self.l, dtype=tf.complex64)
        self.i = tf.constant(1j, dtype=tf.complex64)
        
    def w_algorithm(self, z):
        self.l = tf.cast(self.l, z.dtype)
        self.i = tf.cast(self.i, z.dtype)
        self.a = tf.cast(self.a, z.dtype)
        
        iz = self.i * z
        lp_iz = self.l + iz
        ln_iz = self.l - iz
        
        z_ = lp_iz / ln_iz
        p = polyval(z_, self.a)
        return 2.0 * p / tf.pow(ln_iz, 2) + (1.0 / tf.sqrt(tf.cast(np.pi, tf.complex64))) / ln_iz

    def __call__(self, z):
        sign_r = tf.sign(tf.math.real(z))
        sign_i = tf.sign(tf.math.imag(z))
        z_abs = tf.complex(tf.abs(tf.math.real(z)), tf.abs(tf.math.imag(z)))
        w = self.w_algorithm(z_abs * self.i)
        out = -tf.exp(tf.math.log(w) - tf.pow(z_abs, 2)) + 1.0
        return tf.complex(tf.math.real(out) * sign_r, tf.math.imag(out) * sign_i)


class TRAC(optimizer.Optimizer):
    def __init__(
        self,
        optimizer,
        epsilon=1e-8,
        betas=(0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
        num_coefs=128,
        s_prev=1e-8,
        name="trac",
    ):
        super().__init__(learning_rate=1.,name=name)
        self.optimizer = optimizer
        self.serialized_optimizer = tf.keras.optimizers.serialize(optimizer)
        self.epsilon = epsilon
        self.betas = betas
        self.num_coefs = num_coefs
        self.s_prev = s_prev
        
        self.erf = ERF1994(num_coefs=num_coefs)
        self.f_term = self.s_prev / self.erf_imag(1.0 / tf.sqrt(2.0))
    
    def state_dict(self):
        state_dict = dict()
        return self.optimizer.save_own_variables(state_dict)
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_own_variables(state_dict)
    
    def reset(self):
        self._iterations.assign(0)
        self.betas_tensor = tf.Variable(self.betas)
        self.s = tf.Variable(tf.zeros(()))
        self.h = tf.Variable(tf.zeros(()))
        self.variance = tf.Variable(tf.zeros(len(self.betas)))
        self.sigma = tf.Variable(tf.fill(len(self.betas), 1e-8))
        self._track_variable(self.betas_tensor)
        self._track_variable(self.s)
        self._track_variable(self.h)
        self._track_variable(self.variance)
        self._track_variable(self.sigma)
        for var in self._trainable_variables:
            self.theta_ref[self._get_variable_index(var)].assign(var)
            self.backup_params[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="backup_params"
                                                    )
            self.backup_grads[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="backup_grads"
                                                    )
            self.deltas[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="deltas"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.betas_tensor = tf.Variable(self.betas)
        self.s = tf.Variable(tf.zeros(()))
        self.h = tf.Variable(tf.zeros(()))
        self.variance = tf.Variable(tf.zeros(len(self.betas)))
        self.sigma = tf.Variable(tf.fill(len(self.betas), 1e-8))
        self._track_variable(self.betas_tensor)
        self._track_variable(self.s)
        self._track_variable(self.h)
        self._track_variable(self.variance)
        self._track_variable(self.sigma)
        self.theta_ref = []
        self.backup_params = []
        self.backup_grads = []
        self.deltas = []
        for var in var_list:
            self.theta_ref.append(tf.Variable(var))
            self._track_variable(self.theta_ref[-1])
            self.backup_params.append(self.add_variable_from_reference(
                                reference_variable=var, name="backup_params"
                                                    ))
            self.backup_grads.append(self.add_variable_from_reference(
                                reference_variable=var, name="backup_grads"
                                                    ))
            self.deltas.append(self.add_variable_from_reference(
                                reference_variable=var, name="deltas"
                                                    ))
    
    def erf_imag(self, x):
        if not x.dtype.is_floating:
            x = tf.cast(tf.math.real(x), tf.float32)

        ix = tf.complex(tf.zeros_like(x), x)

        return tf.math.imag(self.erf(ix))
    
    def trac_step(self):
        s = self.s
        h = self.h
        for p in self._trainable_variables:
            theta_ref = self.theta_ref[self._get_variable_index(p)]
            update = self.backup_params[self._get_variable_index(p)]

            self.deltas[self._get_variable_index(p)].assign((update - theta_ref) / (s + self.epsilon))
            update.assign(p - update)

            grad, delta = self.backup_grads[self._get_variable_index(p)], self.deltas[self._get_variable_index(p)]

            product = tf.tensordot(tf.reshape(delta, [-1]), tf.reshape(grad, [-1]), axes=1)
            h.assign_add(product)

            delta.assign_add(update)

            p.assign(theta_ref)

        betas = self.betas_tensor
        variance = self.variance
        sigma = self.sigma

        variance.assign(variance * tf.pow(betas, 2) + tf.pow(h, 2))
        sigma.assign(sigma * betas - h)

        term = self.erf_imag(sigma / (tf.sqrt((2.0 * variance)) + self.epsilon)) * self.f_term
        s.assign(tf.reduce_sum(term))

        scale = tf.maximum(s, 0.0)

        for p in self._trainable_variables:
            p.assign_add(self.deltas[self._get_variable_index(p)] * scale)
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)
    
    def apply_gradients(self, grads_and_vars, tape=None, loss=None):
        self.tape = tape
        self.loss = loss
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations

    def update_step(self, grads, trainable_variables, learning_rate):
        # TODO: backup is first to get the delta of param and grad, but it does not work.
        if self.tape is None and self.loss is None:
            self.optimizer.apply_gradients(zip(grads, trainable_variables))
        elif self.tape is not None:
            self.optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        else:
            self.optimizer.apply_gradients(zip(grads, trainable_variables), self.loss)

        for p, grad in zip(trainable_variables, grads):
            self.backup_params[self._get_variable_index(p)].assign(p)
            self.backup_grads[self._get_variable_index(p)].assign(grad)

        self.trac_step()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "serialized_optimizer": self.serialized_optimizer,
                "epsilon": self.epsilon,
                "betas": self.betas,
                "num_coefs": self.num_coefs,
                "s_prev": self.s_prev,
                "erf": self.erf,
                "f_term": self.f_term,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass