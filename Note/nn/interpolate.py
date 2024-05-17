import tensorflow as tf

def interpolate(input, scale_factor=None, mode='nearest'):
    return tf.image.resize(input, [int(scale_factor[0]), int(scale_factor[1])], method=mode)