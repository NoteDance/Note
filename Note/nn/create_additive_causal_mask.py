import tensorflow as tf

def create_additive_causal_mask(N, dtype = tf.float32):
    indices = tf.range(N)
    mask = indices[:, None] < indices[None]
    # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
    # TODO: Should replace this with finfo(dtype).min
    mask = tf.cast(mask, dtype) * -1e9
    return mask