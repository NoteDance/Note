import tensorflow as tf


class alpha_dropout:
    """Applies Alpha Dropout to the input.

    Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
    to their original values, in order to ensure the self-normalizing property
    even after this dropout.
    Alpha Dropout fits well to Scaled Exponential Linear Units
    by randomly setting activations to the negative saturation value.

    Args:
      rate: float, drop probability (as with `Dropout`).
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

    def __init__(self, rate, noise_shape=None, seed=7):
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        return self.noise_shape if self.noise_shape else tf.shape(inputs)

    def __call__(self, inputs, train_flag=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs(inputs=inputs, rate=self.rate):
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                alpha_p = -alpha * scale

                kept_idx = tf.math.greater_equal(tf.random.uniform(noise_shape), rate)
                kept_idx = tf.cast(kept_idx, inputs.dtype)

                # Get affine transformation params
                a = ((1 - rate) * (1 + rate * alpha_p**2)) ** -0.5
                b = -a * alpha_p * rate

                # Apply mask
                x = inputs * kept_idx + alpha_p * (1 - kept_idx)

                # Do affine transformation
                return a * x + b

            return tf.cond(train_flag, lambda: dropped_inputs, lambda: inputs)
        return inputs