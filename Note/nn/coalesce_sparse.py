import tensorflow as tf


def coalesce_sparse(sp: tf.SparseTensor) -> tf.SparseTensor:
    dense_shape = tf.cast(sp.dense_shape, tf.int64)
    multipliers = tf.concat([
        tf.math.cumprod(dense_shape[1:], reverse=False),
        tf.constant([1], dtype=tf.int64)
    ], axis=0)
    linear_idx = tf.reduce_sum(sp.indices * multipliers, axis=1)
    unique_idx, segment_ids = tf.unique(linear_idx)
    summed_vals = tf.math.unsorted_segment_sum(
        sp.values, segment_ids, tf.shape(unique_idx)[0]
    )
    unraveled = tf.unravel_index(unique_idx, sp.dense_shape)
    new_indices = tf.stack(unraveled, axis=1)
    return tf.SparseTensor(new_indices, summed_vals, sp.dense_shape)