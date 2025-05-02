import tensorflow as tf


def softplus(x, beta=1.0, threshold=20.0):
    if beta != 1.0:
        x = x * beta
    x = tf.where(
        x > threshold,
        x,
        tf.math.log(1 + tf.exp(x))
    )
    if beta != 1.0:
        x = x / beta
    return x