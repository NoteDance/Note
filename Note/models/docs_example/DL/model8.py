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
    def __init__(self, input_dim: int, n_train_samples: int, trained_param):
        self.Model_trained = Model_trained(input_dim, n_train_samples)
        nn.assign_param(self.Model_trained.param, trained_param)
        self.Model_new = Model_new(input_dim, n_train_samples + 1)
        nn.assign_param(self.Model_new.param[:-2], trained_param[:-2])
        self.Model_new.param[-2][:, :-1].assign(trained_param[-2])
        self.Model_new.param[-1][:-1].assign(trained_param[-1])
        self.new_class_batch_size = 64
        self.param = [self.Model_new.param, [self.Model_new.param[-2][:, -1:], self.Model_new.param[-1][-1:]]]

    # ------------------------------------------------------------------
    def __call__(self, x):
        # Old-class samples → knowledge distillation
        self.distribution_trained = self.Model_trained(x[self.new_class_batch_size:])
        self.distribution_new     = self.Model_new(x[self.new_class_batch_size:])
        # New-class samples → main task output (loss comes only from here)
        return self.Model_new(x[:self.new_class_batch_size])

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

        kl = tf.reduce_mean(
            tf.reduce_sum(
                p * (tf.math.log(p + 1e-8) - tf.math.log(q + 1e-8)),
                axis=-1
            )
        )
        kl_weight = tf.stop_gradient(kl)                           # pure scaling factor, no gradient

        # ----------------------------------------------------------------
        # Parameter difference norm penalty (gradients flow only through Model_new.param)
        # d3 is truncated to align with the old-class portion
        # ----------------------------------------------------------------
        param_penalty = tf.constant(0.0, dtype=tf.float32)
        n = len(self.Model_trained.param)

        for i, (p_t, p_n) in enumerate(
            zip(self.Model_trained.param, self.Model_new.param)
        ):
            p_t = tf.cast(tf.stop_gradient(p_t), tf.float32)
            p_n = tf.cast(p_n, tf.float32)

            if i == n - 2:      # d3 weight [64,10] vs [64,11]
                diff = p_n[:, :10] - p_t
            elif i == n - 1:    # d3 bias   [10]    vs [11]
                diff = p_n[:10] - p_t
            else:
                diff = p_n - p_t

            param_penalty = param_penalty + (
                tf.norm(diff, ord='fro')
                if len(diff.shape) > 1
                else tf.norm(diff, ord=2)
            )

        return [loss + self.lambda_param * kl_weight * param_penalty, kl]

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
