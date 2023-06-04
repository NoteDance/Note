import tensorflow as tf


def layer_normalization(inputs,epsilon=1e-6):
    mean,variance=tf.nn.moments(inputs,axes=[-1],keepdims=True)
    std=tf.math.sqrt(variance)
    normalized=(inputs-mean)/(std+epsilon)
    return normalized