import tensorflow as tf


def narrow(tensor, dim, start, size):
    rank = tf.rank(tensor)
    shape = tf.shape(tensor)
    dim = tf.where(dim < 0, dim + rank, dim)
    before = tf.zeros([dim], dtype=tf.int32)
    after = tf.zeros([rank - dim - 1], dtype=tf.int32)
    begin = tf.concat([before, tf.expand_dims(start, 0), after], axis=0)
    one_hot = tf.one_hot(dim, rank, dtype=tf.int32)
    size_for_tf_slice = (shape * (1 - one_hot)) + size * one_hot
    return tf.slice(tensor, begin, size_for_tf_slice)
