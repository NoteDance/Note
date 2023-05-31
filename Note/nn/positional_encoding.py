import tensorflow as tf


def positional_encoding(max_len,d_model):
    pos_enc=tf.zeros((max_len,d_model))
    angles=tf.zeros((max_len,d_model))
    pos=tf.range(max_len)[:,tf.newaxis]
    i=tf.range(d_model)[tf.newaxis,:]
    even_mask=i%2==0
    odd_mask=~even_mask
    angles=tf.where(even_mask,tf.math.sin(pos/(10000**(i/d_model))),angles)
    angles=tf.where(odd_mask,tf.math.cos(pos/(10000**((i-1)/d_model))),angles)
    pos_enc=angles
    return pos_enc