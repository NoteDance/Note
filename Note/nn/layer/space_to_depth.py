import tensorflow as tf


class SpaceToDepth:
    def __init__(self, block_size=4):
        assert block_size == 4
        self.bs = block_size

    def __call__(self, x):
        N, H, W, C = x.shape
        x = tf.reshape(x, (N, H // self.bs, self.bs, W // self.bs, self.bs, C))  # (N, H//bs, bs, W//bs, bs, C)
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))  # (N, bs, bs, C, H//bs, W//bs)
        x = tf.reshape(x, (N, H // self.bs, W // self.bs, C * self.bs * self.bs))  # (N, H//bs, W//bs, C*bs^2)
        return x


class DepthToSpace:

    def __init__(self, block_size):
        self.bs = block_size

    def __call__(self, x):
        N, H, W, C = x.shape
        x = tf.reshape(x, (N, H, W, self.bs, self.bs, C // (self.bs ** 2)))  # (N, H, W, bs, bs, C//bs^2)
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))  # (N, H, bs, W, bs, C//bs^2)
        x = tf.reshape(x, (N, H * self.bs, W * self.bs, C // (self.bs ** 2)))  # (N, H * bs, W * bs, C//bs^2)
        return x