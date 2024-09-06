import tensorflow as tf
from Note import nn


class gaussian_dropout:
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Args:
      rate: Float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, rate, seed=7):
        self.rate = rate
        self.seed = seed
        self.random_generator = tf.random.Generator.from_seed(self.seed)
        self.train_flag = True
        nn.Model.layer_list.append(self)
        if nn.Model.name!=None and nn.Model.name not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name]=[]
            nn.Model.layer_eval[nn.Model.name].append(self)
        elif nn.Model.name!=None:
            nn.Model.layer_eval[nn.Model.name].append(self)

    def __call__(self, data, training=None):
        if training==None:
            training=self.train_flag
        if 0 < self.rate < 1:
            def noised():
                stddev = tf.math.sqrt(self.rate / (1.0 - self.rate))
                return data * self.random_generator.normal(
                    shape=tf.shape(data),
                    mean=1.0,
                    stddev=stddev,
                    dtype=data.dtype,
                )

            return tf.cond(training, noised, lambda: data)
        return data
