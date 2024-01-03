import tensorflow as tf

def pairwise_distance(x, y, p=2, eps=1e-6, keepdim=False):
  diff = tf.math.subtract(x, y) + eps
  norm = tf.math.reduce_sum(tf.math.abs(diff ** p), axis=-1) ** (1/p)
  if keepdim==True:
      norm = tf.expand_dims(norm, -1)
  return norm