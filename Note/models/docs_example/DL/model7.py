from Note import nn
import tensorflow as tf


class Model(nn.Model):
    def __init__(self, input_dim: int, n_train_samples: int):
        super().__init__()
        self.d1 = nn.dense(128, input_dim, activation='relu')
        self.d2 = nn.dense(64,  128,       activation='relu')
        self.d3 = nn.dense(10,  64)

        self.pr_flag = tf.Variable(False, trainable=False, dtype=tf.bool,
                                   name='pr_flag')
        self.ess     = tf.Variable(0.0,   trainable=False, dtype=tf.float32,
                                   name='ess')
        self.max_ess = tf.constant(float(n_train_samples), dtype=tf.float32)

        self.param_copy = [
                        tf.Variable(tf.zeros_like(p), trainable=False,
                                    name=f'param_copy_{i}')
                        for i, p in enumerate(self.param)
                    ]
        self.svd_k = 7

    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
    
    def compute_svd_penalty(self) -> tf.Tensor:
        weight  = 1.0 - tf.minimum(self.ess / self.max_ess, 1.0)
        penalty = tf.constant(0.0, dtype=tf.float32)

        for p, pc in zip(self.param, self.param_copy):
            shape = p.shape
            if len(shape) < 2:
                continue

            rows = 1
            for d in shape[:-1]:
                rows *= d
            cols    = shape[-1]
            p_2d    = tf.reshape(tf.cast(p,  tf.float32), [rows, cols])
            pc_2d   = tf.reshape(tf.cast(pc, tf.float32), [rows, cols])

            k = tf.minimum(self.svd_k, tf.minimum(rows, cols))

            s_param, u_param, v_param = tf.linalg.svd(p_2d,  full_matrices=False)
            s_copy, u_copy,  v_copy = tf.linalg.svd(pc_2d, full_matrices=False)

            u_param = u_param[:, :k]    # [rows, k]
            u_copy  = u_copy[:,  :k]    # [rows, k]
            s_param = u_param[:, :k]    # [rows, k]
            s_copy  = u_copy[:,  :k]    # [rows, k]
            v_param = u_param[:, :k]    # [rows, k]
            v_copy  = u_copy[:,  :k]    # [rows, k]
            
            approx_param = tf.matmul(u_param, tf.matmul(tf.linalg.diag(s_param), v_param, adjoint_b=True))
            approx_copy = tf.matmul(u_copy, tf.matmul(tf.linalg.diag(s_copy), v_copy, adjoint_b=True))

            diff_norm = tf.reduce_mean((approx_param - approx_copy)**2)
            penalty = penalty + diff_norm

        return weight * penalty
    
    def loss_func(self, loss):
        penalty = tf.cond(
            self.pr_flag,
            true_fn  = lambda: self.compute_svd_penalty(),
            false_fn = lambda: tf.constant(0.0, dtype=tf.float32)
        )
        return loss + penalty
