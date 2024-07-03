import tensorflow as tf
from Note import nn


class spatial_dropout3d:
    """Spatial 3D dropout layer.

    This layer randomly sets 3D feature maps along the last three dimensions to zero with a
    frequency of `rate` at each step during training time in order to prevent overfitting.
    Inputs not set to zero are scaled up by 1/(1 - rate) such that the sum over all inputs
    is unchanged.

    Arguments:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        seed: A Python integer to use as random seed.

    Call arguments:
        data: A 5D tensor.
        train_flag: A Python boolean indicating whether to apply dropout to the inputs or not. 
            If True, the layer will randomly set 3D feature maps to zero with a frequency of rate. 
            If False, the layer will return the inputs unchanged.

    References:
        - Efficient Object Localization Using Convolutional Networks
    """

    def __init__(self, rate, seed=None):
        self.rate = rate
        self.seed = seed
        self.train_flag = True
        nn.Model.layer_list.append(self)
        if nn.Model.name_!=None and nn.Model.name_ not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name_]=[]
            nn.Model.layer_eval[nn.Model.name_].append(self)
        elif nn.Model.name_!=None:
            nn.Model.layer_eval[nn.Model.name_].append(self)

    def __call__(self, data, training=None):
        if training==None:
            training=self.train_flag
        def dropped_inputs():
            # Generate a mask with shape (batch_size, 1, 1, 1, channels)
            noise_shape = (tf.shape(data)[0], 1, 1, 1, tf.shape(data)[4])
            mask = tf.random.stateless_binomial(noise_shape, seed=[self.seed, 0], counts=1, probs=(1 - self.rate), 
                                                output_dtype=data.dtype)
            # Scale up the input by 1/(1 - rate) and apply the mask
            return data * mask * (1.0 / (1.0 - self.rate))

        return tf.cond(training, dropped_inputs, lambda: data)
