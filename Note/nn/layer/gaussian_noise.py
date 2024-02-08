import tensorflow as tf


class gaussian_noise:
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    Args:
      stddev: Float, standard deviation of the noise distribution.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, stddev, seed=7):
        self.stddev = stddev
        self.seed = seed
        self.random_generator = tf.random.Generator.from_seed(self.seed)

    def __call__(self, data, train_flag=True):
        def noised():
            return data + self.random_generator.normal(
                                shape=tf.shape(data),
                                mean=0.0,
                                stddev=self.stddev,
                                dtype=data.dtype,
                            )

        return tf.cond(train_flag, lambda: noised(), lambda: data)