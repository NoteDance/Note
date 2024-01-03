import tensorflow as tf
import numpy as np

def cosine_similarity(x1, x2, axis=1, eps=1e-8):
    w12 = tf.reduce_sum(tf.multiply(x1, x2), axis=axis)
    w1 = tf.reduce_sum(tf.multiply(x1, x1), axis=axis)
    w2 = tf.reduce_sum(tf.multiply(x2, x2), axis=axis)
    n12 = tf.sqrt(clip(w1 * w2, eps * eps))
    cos_sim = w12 / n12
    return cos_sim

def clip(x, min):
    x_dtype = x.dtype
    if x_dtype == tf.int32:
        max = np.iinfo(np.int32).max - 2**7
    elif x_dtype == tf.int64:
        max = np.iinfo(np.int64).max - 2**39
    elif x_dtype == tf.float16:
        max = float(np.finfo(np.float16).max)
    else:
        max = float(np.finfo(np.float32).max)

    return tf.clip_by_value(x, min, max)