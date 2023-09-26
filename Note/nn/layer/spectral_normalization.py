import tensorflow as tf
from Note.nn.initializer import initializer


class spectral_normalization:
    """Performs spectral normalization on the weights of a target layer.

    This wrapper controls the Lipschitz constant of the weights of a layer by
    constraining their spectral norm, which can stabilize the training of GANs.

    Args:
      layer: A `keras.layers.Layer` instance that
        has either a `kernel` (e.g. `Conv2D`, `Dense`...)
        or an `embeddings` attribute (`Embedding` layer).
      power_iterations: int, the number of iterations during normalization.

    Reference:

    - [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1):
        self.layer=layer
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero. Received: "
                f"`power_iterations={power_iterations}`"
            )
        self.power_iterations = power_iterations
        if hasattr(self.layer, "weight"):
            self.kernel = self.layer.weight
        elif hasattr(self.layer, "embeddings"):
            self.kernel = self.layer.embeddings
        else:
            raise ValueError(
                f"{type(self.layer).__name__} object has no attribute 'kernel' "
                "nor 'embeddings'"
            )

        self.kernel_shape = self.kernel.shape.as_list()

        self.vector_u = initializer(
            shape=(1, self.kernel_shape[-1]),
            initializer=['truncated_normal',0.02],
            dtype=self.kernel.dtype,
        )
        
        self.train_flag=False

    def output(self, data, train_flag=False):
        if train_flag:
            self.normalize_weights()

        output = self.layer.output(data)
        return output

    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.kernel` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        weights = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        vector_u = self.vector_u

        # check for zeroes weights
        if not tf.reduce_all(tf.equal(weights, 0.0)):
            for _ in range(self.power_iterations):
                vector_v = tf.math.l2_normalize(
                    tf.matmul(vector_u, weights, transpose_b=True)
                )
                vector_u = tf.math.l2_normalize(tf.matmul(vector_v, weights))
            vector_u = tf.stop_gradient(vector_u)
            vector_v = tf.stop_gradient(vector_v)
            sigma = tf.matmul(
                tf.matmul(vector_v, weights), vector_u, transpose_b=True
            )
            self.vector_u.assign(tf.cast(vector_u, self.vector_u.dtype))
            self.kernel.assign(
                tf.cast(
                    tf.reshape(self.kernel / sigma, self.kernel_shape),
                    self.kernel.dtype,
                )
            )