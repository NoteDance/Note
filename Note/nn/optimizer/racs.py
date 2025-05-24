""" RACS
https://arxiv.org/abs/2502.07752

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class RACS(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        alpha=0.05,
        gamma=1.01,
        maximize=False,
        weight_decouple=True,
        fixed_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="racs",
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
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.maximize = maximize
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.s[self._get_variable_index(var)].assign(tf.zeros(var.shape[0], dtype=var.dtype))
            self.q[self._get_variable_index(var)].assign(tf.ones(var.shape[1], dtype=var.dtype))
            self.theta[self._get_variable_index(var)].assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.s = []
        self.q = []
        self.theta = []
        for var in var_list:
            self.s.append(
               tf.Variable(tf.zeros(var.shape[0], dtype=var.dtype))
            )
            self.q.append(
                tf.Variable(tf.ones(var.shape[1], dtype=var.dtype))
            )
            self.theta.append(
                tf.Variable(tf.zeros((), dtype=var.dtype))
            )
            self._track_variable(self.s[-1])
            self._track_variable(self.q[-1])
            self._track_variable(self.theta[-1])

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'RACS does not support sparse gradient.')
        
        if variable.dtype.is_complex:
            raise RuntimeError(
                'RACS does not support complex parameter.')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        if len(gradient.shape) < 2:
            gradient = tf.reshape(gradient, (len(gradient), 1))
        elif len(gradient.shape) > 2:
            gradient = tf.reshape(gradient, (len(gradient), -1))
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        s = self.s[self._get_variable_index(variable)]
        q = self.q[self._get_variable_index(variable)]
        theta = self.theta[self._get_variable_index(variable)]
        
        grad_p2 = tf.pow(gradient, 2)
        s.assign(s * self.beta + tf.reduce_mean(grad_p2, axis=1) * (1.0 - self.beta))
        q.assign(q * self.beta + tf.reduce_mean(grad_p2, axis=0) * (1.0 - self.beta))
    
        s_sq = tf.expand_dims(tf.sqrt(s + self.epsilon), axis=1)
        q_sq = tf.expand_dims(tf.sqrt(q + self.epsilon), axis=0)
        
        grad_hat = gradient / (s_sq * q_sq)
        
        grad_hat_norm = tf.norm(grad_hat)
        def true_fn():
            return self.gamma / tf.maximum(grad_hat_norm / (theta + self.epsilon), self.gamma)
        def false_fn():
            return 1.0
        threshold = tf.cond(step > 1, true_fn, false_fn)
        theta.assign(grad_hat_norm * threshold)
        
        variable.assign_add(tf.reshape(grad_hat, variable.shape) * -lr * self.alpha * threshold)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "maximize": self.maximize,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class Alice(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.02,
        betas=(0.9, 0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.0,
        alpha=0.3,
        alpha_c=0.4,
        update_interval=200,
        rank=256,
        gamma=1.01,
        leading_basis=40,
        maximize=False,
        weight_decouple=True,
        fixed_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="alice",
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
        self.alpha = alpha
        self.alpha_c = alpha_c
        self.update_interval = update_interval
        self.rank = rank
        self.gamma = gamma
        self.leading_basis = leading_basis
        self.maximize = maximize
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            m, n = var.shape
            self.U[self._get_variable_index(var)].assign(tf.zeros((m, self.rank), dtype=var.dtype))
            self.Q[self._get_variable_index(var)].assign(tf.zeros((self.rank, self.rank), dtype=var.dtype))
            self.m[self._get_variable_index(var)].assign(tf.zeros((self.rank, n), dtype=var.dtype))
            self.v[self._get_variable_index(var)].assign(tf.zeros((self.rank, n), dtype=var.dtype))
            self.p[self._get_variable_index(var)].assign(tf.zeros((n,), dtype=var.dtype))
            self.phi[self._get_variable_index(var)].assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.U = []
        self.Q = []
        self.m = []
        self.v = []
        self.p = []
        self.phi = []
        for var in var_list:
            m, n = var.shape
            self.U.append(
               tf.Variable(tf.zeros((m, self.rank), dtype=var.dtype))
            )
            self.Q.append(
                tf.Variable(tf.zeros((self.rank, self.rank), dtype=var.dtype))
            )
            self.m.append(
                tf.Variable(tf.zeros((self.rank, n), dtype=var.dtype))
            )
            self.v.append(
                tf.Variable(tf.zeros((self.rank, n), dtype=var.dtype))
            )
            self.p.append(
                tf.Variable(tf.zeros((n,), dtype=var.dtype))
            )
            self.phi.append(
                tf.Variable(tf.zeros((), dtype=var.dtype))
            )
            self._track_variable(self.U[-1])
            self._track_variable(self.Q[-1])
            self._track_variable(self.m[-1])
            self._track_variable(self.v[-1])
            self._track_variable(self.p[-1])
            self._track_variable(self.phi[-1])
    
    @staticmethod
    def subspace_iteration(
        a, mat, num_steps = 1
    ):
        r"""Perform subspace iteration."""
        u = mat
        for _ in range(num_steps):
            u, _ = tf.linalg.qr(tf.matmul(a, u))

        return tf.linalg.eigh(tf.matmul(tf.matmul(tf.transpose(u), a), u))

    def switch(self, q, u_prev, rank, leading_basis):
        vals, vecs = self.subspace_iteration(tf.cast(q, tf.float32), tf.cast(u_prev, tf.float32), num_steps=1)

        leading_indices = tf.argsort(vals, direction='DESCENDING')[:leading_basis]
        u_t1 = tf.gather(vecs, leading_indices, axis=1)

        u_c, _ = tf.linalg.qr(tf.eye(q.shape[0], dtype=tf.float32) - tf.matmul(u_t1, u_t1, transpose_b=True))
        u_t2 = u_c[:, :rank - leading_basis]  # fmt: skip

        return tf.cast(tf.concat([u_t1, u_t2], axis=1), q.dtype)

    @staticmethod
    def compensation(
        grad,
        u,
        p,
        phi,
        gamma,
        decay_rate,
        rank,
    ):
        m, n = grad.shape

        sigma = tf.matmul(tf.transpose(u), grad)
        
        p.assign(tf.maximum(p * decay_rate + (tf.reduce_sum(tf.pow(grad, 2), axis=0) - tf.reduce_sum(tf.pow(sigma, 2))) * 1.0 - decay_rate, 1e-8))

        d = tf.zeros_like(grad)
        diag_len = min(m, n)
        diag_indices = tf.range(diag_len)
        p_sqrt_inv = 1.0 / tf.sqrt(p[:diag_len])
        indices = tf.stack([diag_indices, diag_indices], axis=1)
        d = tf.tensor_scatter_nd_update(d, indices, p_sqrt_inv)

        c_t = math.sqrt(m - rank) * (grad - tf.matmul(u, sigma)) * d if m >= rank else tf.zeros_like(grad)
        
        def true_fn():
            return gamma / tf.maximum(tf.norm(c_t) / phi, gamma)
        def false_fn():
            return tf.ones_like(phi)
        n = tf.cond(phi > 0, true_fn, false_fn)

        c_t *= n
        phi = tf.norm(c_t)

        return c_t, phi

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'RACS does not support sparse gradient.')
        
        if variable.dtype.is_complex:
            raise RuntimeError(
                'RACS does not support complex parameter.')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2, beta3 = self.betas
        
        if len(gradient.shape) < 2:
            gradient = tf.reshape(gradient, (len(gradient), 1))
        elif len(gradient.shape) > 2:
            gradient = tf.reshape(gradient, (len(gradient), -1))
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        q = self.Q[self._get_variable_index(variable)]
        u = self.U[self._get_variable_index(variable)]
        m = self.m[self._get_variable_index(variable)]
        v = self.v[self._get_variable_index(variable)]
        p = self.p[self._get_variable_index(variable)]
        phi = self.phi[self._get_variable_index(variable)]

        def true_fn(u = u):
            q_t = beta3 * tf.matmul(tf.matmul(u, q), tf.transpose(u)) + (1.0 - beta3) * tf.matmul(gradient, tf.transpose(gradient))
            u = self.switch(q_t, u, self.rank, self.leading_basis)
        def false_fn():
            pass
        tf.cond(tf.logical_or(step == 1, step % self.update_interval == 0), true_fn, false_fn)
        
        sigma = tf.matmul(tf.transpose(u), gradient)
                    
        q.assign(q * beta3 + tf.matmul(sigma, tf.transpose(sigma)) * (1.0 - beta3))
        m.assign(m * beta1 + sigma * (1.0 - beta1))
        v.assign(v * beta2 + tf.pow(sigma, 2) * (1.0 - beta2))
    
        c_t, phi = self.compensation(gradient, u, p, phi, self.gamma, beta1, self.rank)
        
        update = tf.matmul(u, (m / tf.sqrt(v)))
        update += c_t * self.alpha_c
        
        variable.assign_add(tf.reshape(update, variable.shape) * -lr * self.alpha)
        
        phi.assign(phi)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "alpha_c": self.alpha_c,
                "update_interval": self.update_interval,
                "rank": self.rank,
                "gamma": self.gamma,
                "leading_basis": self.leading_basis,
                "maximize": self.maximize,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass