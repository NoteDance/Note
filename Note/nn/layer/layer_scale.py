import tensorflow as tf
from Note import nn


class LayerScale(nn.Layer):
    """ LayerScale on tensors with channels in last-dim.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * tf.ones(dim))

    def __call__(self, x):
        return x * self.gamma


class LayerScale2d(nn.Layer):
    """ LayerScale for tensors with torch 2D NHWC layout.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
    ):
        super().__init__()
        self.gamma = nn.Parameter(init_values * tf.ones(dim))

    def __call__(self, x):
        gamma = tf.reshape(self.gamma, (1, 1, 1, -1))
        return x * gamma
