from Note import nn
import tensorflow as tf


class Model_new(nn.Model):
    """Expanded model for class-incremental learning (11 classes)."""
    def __init__(self, input_dim: int, n_train_samples: int):
        super().__init__()
        self.d1 = nn.dense(128, input_dim, activation='relu')
        self.d2 = nn.dense(64,  128,       activation='relu')
        self.d3 = nn.dense(11,  64)

    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


class Model(nn.Model):
    """
    Class-Incremental Learning Wrapper.
    Combines Knowledge Distillation, SVD Parameter Regularization,
    and Adaptive Lambda Annealing based on Cosine Decay + KL Exponential Moving Average (EMA).
    """
    def __init__(self, input_dim: int, n_train_samples: int, trained_param, distributions, old_train_data, update_freq, kl_threshold_, kl_threshold: float,
                 z = 0, u = 1, lambda_param = 0.25):
        super().__init__()
        self.old_train_data = old_train_data
        self.distributions = distributions
        
        # Instantiate frozen baseline model and transfer pre-trained parameters
        self.trained_param = trained_param
        
        # Instantiate new model and inherit pre-trained parameters
        self.Model_new = Model_new(input_dim, n_train_samples + 1)
        nn.assign_param(self.Model_new.param[:-2], trained_param[:-2])
        self.Model_new.param[-2][:, :-1].assign(trained_param[-2])  # Copy d3 weight slice (first 10 outputs)
        self.Model_new.param[-1][:-1].assign(trained_param[-1])     # Copy d3 bias slice (first 10 outputs)
        
        self.new_class_batch_size = 64
        self.param = [self.Model_new.param, [self.Model_new.param[-2][:, -1:], self.Model_new.param[-1][-1:]]]
        self.init_priority = 1.0
        self.svd_k = tf.Variable(7)
        self.sv_threshold = 1e-7
        
        self.update_freq = update_freq
        self.kl_mean = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.kl_threshold_ = kl_threshold_
        
        # Thresholds & hyper-parameters
        self.kl_threshold = tf.constant(kl_threshold, dtype=tf.float32)
        
        self.z = [tf.Variable(0, trainable=False, dtype=tf.float32) for _ in len(self.trained_param)]
        self.u = [tf.Variable(1, trainable=False, dtype=tf.float32) for _ in len(self.trained_param)]
        self.lambda_param = lambda_param
        self.diff_norm_old = tf.Variable(0, trainable=False, dtype=tf.float32)

    # ------------------------------------------------------------------
    def __call__(self, x):
        # Sample replay buffer containing old class data
        old_data, self.distribution_trained = self.prioritized_replay.sample(self.old_train_data, self.distributions, self.alpha, self.pr_batch_size)
        
        # Forward pass for knowledge distillation
        self.distribution_new = self.Model_new(old_data)
        
        # Main task forward pass on new data batch
        return self.Model_new(x)

    def update_z_u(self, diff_norm, i) -> None:
        self.z[i].assign(self.z[i] + 2 * self.u[i] * diff_norm)
        self.u[i].assign(tf.cond(diff_norm < self.lambda_param * self.diff_norm_old or self.diff_norm_old == 0, lambda: self.u[i], lambda: 2 * self.u[i]))

    # ------------------------------------------------------------------
    # Loss Function Computation
    # ------------------------------------------------------------------
    def loss_func(self, loss: tf.Tensor) -> tf.Tensor:
        # ----------------------------------------------------------------
        # 1. KL Divergence (p_trained || q_new)
        # ----------------------------------------------------------------
        p = tf.nn.softmax(self.distribution_trained)   # [B, 10]
        q_full = tf.nn.softmax(self.distribution_new)  # [B, 11]
        q = q_full[:, :10]                             # Slice old-class logits

        kl_per_sample = tf.reduce_sum(
            p * (tf.math.log(p + 1e-8) - tf.math.log(q + 1e-8)),
            axis=-1
        )  

        kl_per_sample_detached = tf.stop_gradient(kl_per_sample)
        
        if self.pr_batch_size is not None:
            self.prioritized_replay.loss_.assign(kl_per_sample_detached)

        # Gate KL per sample based on threshold
        mask = tf.cast(kl_per_sample_detached > self.kl_threshold, kl_per_sample.dtype)
        gated_kl_per_sample = kl_per_sample_detached + mask * (kl_per_sample - kl_per_sample_detached)

        kl = tf.reduce_mean(gated_kl_per_sample)

        # ----------------------------------------------------------------
        # 2. SVD Parameter Penalty (Regularization)
        # ----------------------------------------------------------------
        n = len(self.trained_param)
        penalty = tf.constant(0.0, dtype=tf.float32)
        
        for i, (p_t, p_n) in enumerate(zip(self.trained_param, self.Model_new.param)):
            p_t = tf.cast(tf.stop_gradient(p_t), tf.float32)
            p_n = tf.cast(p_n, tf.float32)
            
            shape = p_t.shape
            if len(shape) < 2:  # Skip 1D bias vectors
                continue
            
            if i == n - 2:      # Align d3 weights: [64, 10] vs slice [64, 10]
                p_n = p_n[:, :10]
            
            # Reshape 2D matrices for SVD analysis
            rows = 1
            for d in shape[:-1]:
                rows *= d
            cols = shape[-1]
            p_t_2d = tf.reshape(p_t, [rows, cols])
            p_n_2d = tf.reshape(p_n, [rows, cols])

            k = tf.minimum(self.svd_k, tf.minimum(rows, cols))
            
            # Singular Value Decomposition
            s_param, u_param, v_param = tf.linalg.svd(p_t_2d, full_matrices=False)
            s_copy, u_copy, v_copy = tf.linalg.svd(p_n_2d, full_matrices=False)

            # Truncate top-k singular vectors and values
            u_param_k = u_param[:, :k]
            u_copy_k  = u_copy[:, :k]
            s_param_k = s_param[:k]
            s_copy_k  = s_copy[:k]
            v_param_k = v_param[:, :k]
            v_copy_k  = v_copy[:, :k]
            
            s_copy_rest  = s_copy[k:]
            eps = 1e-7
            cond_copy_rest  = tf.reduce_sum(s_copy_rest[0])  / (tf.reduce_sum(s_copy_rest[-1])  + eps)
            
            # Low-rank approximations
            approx_param = tf.matmul(u_param_k, tf.matmul(tf.linalg.diag(s_param_k), v_param_k, adjoint_b=True))
            approx_copy  = tf.matmul(u_copy_k,  tf.matmul(tf.linalg.diag(s_copy_k),  v_copy_k,  adjoint_b=True))
            
            dot_per_col = tf.reduce_sum(approx_param * approx_copy, axis=0)
            dist_param_col = tf.norm(approx_param * approx_param, axis=0)
            dist_copy_col = tf.norm(approx_copy * approx_copy, axis=0)

            # After transforming the original problem into an augmented Lagrangian problem, the purpose of parameter tuning has changed.
            diff_norm = tf.reduce_mean((u_param_k - u_copy_k + self.z[i] / 2 * self.u[i])**2)
            self.update_z_u(diff_norm, i)
            diff_mean = tf.reduce_mean(dot_per_col - dist_param_col * dist_copy_col)
            self.diff_norm_old.assign(diff_norm)

            penalty = penalty + diff_norm + diff_mean

        # Final loss formulation
        return loss + kl + penalty + (cond_copy_rest - 1)**2

    # ------------------------------------------------------------------
    # Polyak / Soft Update for Old Class Parameter Stabilization
    # ------------------------------------------------------------------
    def soft_update(self) -> None:
        """
        Soft update rule: Model_new <- tau * Model_trained + (1 - tau) * Model_new
        Pulls Model_new back toward Model_trained to mitigate catastrophic forgetting.
        """
        n = len(self.trained_param)
        for i, (p_t, p_n) in enumerate(zip(self.trained_param, self.Model_new.param)):
            dtype = p_n.dtype
            p_t_c = tf.cast(p_t, dtype)

            if i == n - 2:      # Output layer weight: pad to [64, 11]
                target = tf.concat([p_t_c, p_n[:, 10:11]], axis=1)
            elif i == n - 1:    # Output layer bias: pad to [11]
                target = tf.concat([p_t_c, p_n[10:11]], axis=0)
            else:               # Hidden layers: update directly
                target = p_t_c

            p_n.assign(self.tau * target + (1.0 - self.tau) * p_n)
    
    def update_param(self):
        if self.batch_counter % self.update_freq == 0 and self.kl_mean <= self.kl_threshold_:
            self.z = [tf.Variable(0, trainable=False, dtype=tf.float32) for _ in len(self.trained_param)]
            self.u = [tf.Variable(1, trainable=False, dtype=tf.float32) for _ in len(self.trained_param)]
            self.diff_norm_old.assign(0)
            self.kl_mean.assign(0.0)
