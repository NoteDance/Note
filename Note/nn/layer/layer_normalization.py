import tensorflow as tf


def layer_normalization(data,epsilon=1e-6):
    mean,variance=tf.nn.moments(data,axes=[-1],keepdims=True)
    std=tf.math.sqrt(variance)
    normalized=(data-mean)/(std+epsilon)
    return normalized