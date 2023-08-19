import tensorflow as tf


def layer_normalization(data,epsilon=1e-6,train_flag=True):
    if train_flag:
        mean,variance=tf.nn.moments(data,axes=[-1],keepdims=True)
        std=tf.math.sqrt(variance)
        normalized=(data-mean)/(std+epsilon)
        return normalized
    else:
        return data
