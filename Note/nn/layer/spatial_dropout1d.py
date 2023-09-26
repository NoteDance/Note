import tensorflow as tf

class spatial_dropout1d:
    """Spatial 1D dropout layer.

    This layer randomly sets 1D feature maps along the last dimension to zero with a
    frequency of `rate` at each step during training time in order to prevent overfitting.
    Inputs not set to zero are scaled up by 1/(1 - rate) such that the sum over all inputs
    is unchanged.

    Arguments:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        seed: A Python integer to use as random seed.

    Call arguments:
        daat: A 3D tensor.
        train_flag: A Python boolean indicating whether to apply dropout to the inputs or not. 
            If True, the layer will randomly set 1D feature maps to zero with a frequency of rate. 
            If False, the layer will return the inputs unchanged.

    References:
        - Efficient Object Localization Using Convolutional Networks
    """

    def __init__(self, rate, seed=7):
        self.rate = rate
        self.seed = seed
        self.train_flag = True

    def output(self, data, train_flag=True):
        self.train_flag = train_flag
        def dropped_inputs():
            # Generate a mask with shape (batch_size, 1, channels)
            noise_shape = (tf.shape(data)[0], 1, tf.shape(data)[2])
            mask = tf.random.stateless_binomial(noise_shape, seed=[self.seed, 0], counts=1, probs=(1 - self.rate), 
                                                output_dtype=data.dtype)
            # Scale up the input by 1/(1 - rate) and apply the mask
            return data * mask * (1.0 / (1.0 - self.rate))

        return tf.cond(train_flag, dropped_inputs, lambda: data)