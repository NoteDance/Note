import tensorflow as tf


def nan_to_num(tensor, nan=0.0, out=None):
    result = tf.where(tf.math.is_nan(tensor), tf.constant(nan, dtype=tensor.dtype), tensor)
    if out is not None:
        out.assign(result)
        return out
    return result