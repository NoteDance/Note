import tensorflow as tf


def flatten(input_tensor):
    input_shape=tf.shape(input_tensor)
    batch_size=input_shape[0]
    num_elements=tf.reduce_prod(input_shape[1:])
    output_tensor=tf.reshape(input_tensor,[batch_size,num_elements])
    return output_tensor
