import tensorflow as tf


def narrow(tensor, dim, start, size):
    if dim < 0:
        dim = tensor.shape.rank + dim
    begin = [0]*dim + [start] + [0]*(tensor.shape.rank - dim - 1)
    size_vec = [-1]*dim + [size] + [-1]*(tensor.shape.rank - dim - 1)
    return tf.slice(tensor, begin, size_vec)