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

    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
    
    
    def compute_svd_penalty(self) -> tf.Tensor:
        weight = 1.0 - tf.minimum(self.ess / self.max_ess, 1.0)

        penalty = tf.constant(0.0, dtype=tf.float32)

        for p, pc in zip(self.param, self.param_copy):
            shape = p.shape
            if len(shape) < 2:
                continue

            rows = 1
            for d in shape[:-1]:
                rows *= d
            cols = shape[-1]

            p_2d  = tf.reshape(tf.cast(p,  tf.float32), [rows, cols])
            pc_2d = tf.reshape(tf.cast(pc, tf.float32), [rows, cols])

            _, u_param, _ = tf.linalg.svd(p_2d,  full_matrices=False)
            _, u_copy,  _ = tf.linalg.svd(pc_2d, full_matrices=False)

            sim_param = tf.matmul(u_param, u_param, transpose_b=True)
            sim_copy  = tf.matmul(u_copy,  u_copy,  transpose_b=True)

            penalty = penalty + tf.norm(sim_param - sim_copy, ord='fro')

        return weight * penalty


    @tf.function(jit_compile=True)
    def train_step(self, train_data, labels, loss_object,
                   train_loss, train_accuracy, optimizer):
        with tf.GradientTape(persistent=True) as tape:
            output = self.__call__(train_data)
            loss   = loss_object(labels, output)
            penalty = tf.cond(
                self.pr_flag,
                true_fn  = lambda: self.compute_svd_penalty(),
                false_fn = lambda: tf.constant(0.0, dtype=tf.float32)
            )
            total_loss = loss + penalty

        if type(optimizer) != list:
            grads = tape.gradient(total_loss, self.param)
            optimizer.apply_gradients(zip(grads, self.param))
        else:
            for i in range(len(optimizer)):
                grads = tape.gradient(total_loss, self.param[i])
                optimizer[i].apply_gradients(zip(grads, self.param[i]))

        train_loss(loss)
        if train_accuracy is not None:
            acc = train_accuracy(labels, output)
            return loss, acc
        return loss, None


    @tf.function
    def train_step_(self, train_data, labels, loss_object,
                    train_loss, train_accuracy, optimizer):
        with tf.GradientTape(persistent=True) as tape:
            output = self.__call__(train_data)
            loss   = loss_object(labels, output)
            penalty = tf.cond(
                self.pr_flag,
                true_fn  = lambda: self.compute_svd_penalty(),
                false_fn = lambda: tf.constant(0.0, dtype=tf.float32)
            )
            total_loss = loss + penalty

        if type(optimizer) != list:
            grads = tape.gradient(total_loss, self.param)
            optimizer.apply_gradients(zip(grads, self.param))
        else:
            for i in range(len(optimizer)):
                grads = tape.gradient(total_loss, self.param[i])
                optimizer[i].apply_gradients(zip(grads, self.param[i]))

        train_loss(loss)
        if train_accuracy is not None:
            acc = train_accuracy(labels, output)
            return loss, acc
        return loss, None