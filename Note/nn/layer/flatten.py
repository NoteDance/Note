import tensorflow as tf


def flatten(input_tensor):
    input_shape=tf.shape(input_tensor)
    num_elements=tf.reduce_prod(input_shape)
    output_tensor=tf.reshape(input_tensor,[1,num_elements])
    return output_tensor