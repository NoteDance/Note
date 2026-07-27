from Note import nn
import tensorflow as tf


class Model_trained(nn.Model):
    def __init__(self, input_dim: int, n_train_samples: int):
        super().__init__()
        self.d1 = nn.dense(128, input_dim, activation='relu')
        self.d2 = nn.dense(64,  128,       activation='relu')
        self.d3 = nn.dense(10,  64)

    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


class Model_new(nn.Model):
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
    def __init__(self, input_dim: int, n_train_samples: int, trained_param, old_train_data, kl_threshold: float):
        self.old_train_data=old_train_data
        self.Model_trained = Model_trained(input_dim, n_train_samples)
        nn.assign_param(self.Model_trained.param, trained_param)
        self.Model_new = Model_new(input_dim, n_train_samples + 1)
        nn.assign_param(self.Model_new.param[:-2], trained_param[:-2])
        self.Model_new.param[-2][:, :-1].assign(trained_param[-2])
        self.Model_new.param[-1][:-1].assign(trained_param[-1])
        self.new_class_batch_size = 64
        self.param = [self.Model_new.param, [self.Model_new.param[-2][:, -1:], self.Model_new.param[-1][-1:]]]
        self.init_priority = 1.0
        self.svd_k = tf.Variable(7)
        self.sv_threshold = 1e-7
        self.kl_threshold = tf.constant(kl_threshold, dtype=tf.float32)

    # ------------------------------------------------------------------
    def __call__(self, x):
        old_data = self.prioritized_replay.sample(self.old_train_data, None, self.alpha, self.pr_batch_size)
        # Old-class samples → knowledge distillation
        self.distribution_trained = self.Model_trained(old_data)
        self.distribution_new = self.Model_new(old_data)
        # New-class samples → main task output (loss comes only from here)
        return self.Model_new(x)

    # ------------------------------------------------------------------
    def loss_func(self, loss: tf.Tensor) -> tf.Tensor:
        # ----------------------------------------------------------------
        # KL(p_trained || q_new)
        #   Both sides are fully stop_gradient → KL does not affect parameter updates at all
        #   Used only as a scaling weight for param_penalty
        #   When distributions are identical: KL = 0 → kl_weight = 0 → penalty = 0  ✓
        # ----------------------------------------------------------------
        p = tf.nn.softmax(
            self.distribution_trained
        )                                                           # [B, 10]

        q_full = tf.nn.softmax(
            self.distribution_new
        )                                                           # [B, 11]
        q  = q_full[:, :10]

        kl_per_sample = tf.reduce_sum(
            p * (tf.math.log(p + 1e-8) - tf.math.log(q + 1e-8)),
            axis=-1
        )                                                      

        kl_per_sample_detached = tf.stop_gradient(kl_per_sample)
        
        if self.pr_batch_size!=None:
            self.prioritized_replay.loss_.assign(kl_per_sample_detached)

        mask = tf.cast(kl_per_sample_detached > self.kl_threshold, kl_per_sample.dtype)
        gated_kl_per_sample = kl_per_sample_detached + mask * (kl_per_sample - kl_per_sample_detached)  # [B]

        kl = tf.reduce_mean(gated_kl_per_sample)
        kl_weight = tf.reduce_mean(kl_per_sample_detached)      # pure scaling factor, no gradient

        # ----------------------------------------------------------------
        # Parameter difference norm penalty (gradients flow only through Model_new.param)
        # d3 is truncated to align with the old-class portion
        # ----------------------------------------------------------------
        n = len(self.Model_trained.param)
        
        penalty = tf.constant(0.0, dtype=tf.float32)

        for i, (p_t, p_n) in enumerate(
            zip(self.Model_trained.param, self.Model_new.param)
        ):
            shape = p.shape
            if len(shape) < 2:
                continue
            
            p_t = tf.cast(tf.stop_gradient(p_t), tf.float32)
            p_n = tf.cast(p_n, tf.float32)
            
            if i == n - 2:      # d3 weight [64,10] vs [64,11]
                p_n = p_n[:, :10]
            
            rows = 1
            for d in shape[:-1]:
                rows *= d
            cols    = shape[-1]
            p_t_2d    = tf.reshape(tf.cast(p_t,  tf.float32), [rows, cols])
            p_n_2d   = tf.reshape(tf.cast(p_n, tf.float32), [rows, cols])

            k = tf.minimum(self.svd_k, tf.minimum(rows, cols))
            
            s_param, u_param, v_param = tf.linalg.svd(p_t_2d,  full_matrices=False)
            s_copy, u_copy,  v_copy = tf.linalg.svd(p_n_2d, full_matrices=False)

            u_param = u_param[:, :k]    # [rows, k]
            u_copy  = u_copy[:,  :k]    # [rows, k]
            s_param = u_param[:, :k]    # [rows, k]
            s_copy  = u_copy[:,  :k]    # [rows, k]
            v_param = u_param[:, :k]    # [rows, k]
            v_copy  = u_copy[:,  :k]    # [rows, k]
            
            approx_param = tf.matmul(u_param, tf.matmul(tf.linalg.diag(s_param), v_param, adjoint_b=True))
            approx_copy = tf.matmul(u_copy, tf.matmul(tf.linalg.diag(s_copy), v_copy, adjoint_b=True))
            dot_per_col = tf.reduce_sum(approx_param * approx_copy, axis=0)
            dist_param_col = tf.norm(approx_param * approx_param, axis=0)
            dist_copy_col = tf.norm(approx_copy * approx_copy, axis=0)

            u_param = u_param[:, :k]    # [rows, k]
            u_copy  = u_copy[:,  :k]    # [rows, k]

            diff_norm = tf.reduce_mean((u_param - u_copy)**2)
            diff_mean = tf.reduce_mean(dot_per_col - dist_param_col * dist_copy_col)

            penalty = penalty + diff_norm + diff_mean

        return loss + kl + self.lambda_param * kl_weight * penalty

    # ------------------------------------------------------------------
    # Soft update: Model_new ← τ · Model_trained + (1-τ) · Model_new
    #
    # Direction: pull Model_new toward Model_trained to prevent forgetting old classes
    # d3: use concatenation instead of truncation to keep the 11th neuron (new class) undisturbed
    #
    #   Effect on weight [64, 11]:
    #     First 10 columns: p_n[:,i] ← τ·p_t[:,i] + (1-τ)·p_n[:,i]  (pulled toward old model)
    #     11th column:      p_n[:,10] ← τ·p_n[:,10] + (1-τ)·p_n[:,10] = p_n[:,10]  (unchanged)
    # ------------------------------------------------------------------
    def soft_update(self) -> None:
        n = len(self.Model_trained.param)
        for i, (p_t, p_n) in enumerate(
            zip(self.Model_trained.param, self.Model_new.param)
        ):
            dtype  = p_n.dtype
            p_t_c  = tf.cast(p_t, dtype)

            if i == n - 2:      # d3 weight: trained[64,10] → pad → [64,11]
                target = tf.concat([p_t_c, p_n[:, 10:11]], axis=1)
            elif i == n - 1:    # d3 bias:   trained[10]    → pad → [11]
                target = tf.concat([p_t_c, p_n[10:11]], axis=0)
            else:               # d1, d2: same shape, update directly
                target = p_t_c

            p_n.assign(self.tau * target + (1.0 - self.tau) * p_n)
    
    def update_param(self):