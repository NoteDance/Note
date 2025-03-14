import tensorflow as tf


def sparse_mask(dense_tensor, mask_sparse):
    indices = mask_sparse.indices  # [N, ndims]
    values = tf.gather_nd(dense_tensor, indices)
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=mask_sparse.dense_shape)