import tensorflow as tf


def global_max_pool2d(data):
    return tf.reduce_max(data,[1,2])