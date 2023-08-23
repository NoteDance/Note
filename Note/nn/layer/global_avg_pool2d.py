import tensorflow as tf


def global_avg_pool2d(data):
    return tf.reduce_mean(data,[1,2])