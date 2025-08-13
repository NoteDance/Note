import tensorflow as tf


def unfold(x, dim, size, step):
    x = tf.convert_to_tensor(x)
    r = tf.rank(x)
    dim_t = tf.convert_to_tensor(dim, dtype=tf.int32)
    dim_norm = tf.where(dim_t >= 0, dim_t, r + dim_t)          # normalized dim in [0, r-1]

    # move target axis to last
    rng = tf.range(r)
    keep_mask = tf.not_equal(rng, dim_norm)
    keep_indices = tf.boolean_mask(rng, keep_mask)            # shape (r-1,)
    perm_to_last = tf.concat([keep_indices, tf.expand_dims(dim_norm, 0)], axis=0)
    x_t = tf.transpose(x, perm=perm_to_last)                  # target axis now last

    # frame on last axis -> shape: prefix_dims + [num_frames, size]
    x_f = tf.signal.frame(x_t, frame_length=size, frame_step=step, axis=-1)

    # re-arrange axes so result matches PyTorch:
    # original_shape[:dim] + (num_frames,) + original_shape[dim+1:] + (size,)
    # After framing, axes indices are:
    #   0 .. prefix_rank-1    (these correspond to original dims except 'dim')
    #   prefix_rank           = num_frames
    #   prefix_rank + 1       = size
    r_int = tf.cast(r, tf.int32)
    prefix_rank = r_int - 1

    # left = [0..dim_norm-1]
    left = tf.range(0, dim_norm)
    # right = [dim_norm .. prefix_rank-1]  (these were originally after the unfolded dim)
    right = tf.range(dim_norm, prefix_rank)
    # perm_back = left + [prefix_rank] + right + [prefix_rank+1]
    perm_back = tf.concat([left,
                           tf.expand_dims(prefix_rank, 0),
                           right,
                           tf.expand_dims(prefix_rank + 1, 0)], axis=0)

    out = tf.transpose(x_f, perm=perm_back)
    return out