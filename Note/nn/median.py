import tensorflow as tf


def median(x, axis=-1, keepdims=False):
    s = tf.sort(x, axis=axis)
    L = tf.shape(s)[axis]
    mid = L // 2
    is_odd = tf.equal(tf.math.floormod(L, 2), 1)

    def _odd():
        return tf.gather(s, mid, axis=axis)

    def _even():
        return tf.gather(s, mid - 1, axis=axis)

    med = tf.cond(is_odd, _odd, _even)
    if keepdims:
        med = tf.expand_dims(med, axis=axis)
    return med