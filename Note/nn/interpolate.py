import tensorflow as tf

def interpolate(input, size=None, scale_factor=None, recompute_scale_factor=False, mode='nearest', align_corners=False, antialias=False):
    # Get input shape
    input_shape = tf.shape(input)
    
    # Compute the new size
    if size is None and scale_factor is not None:
        # Compute new size based on scale_factor
        new_size = tf.cast(input_shape[1:3], tf.float32) * scale_factor
    elif size is not None:
        # Use provided size
        new_size = tf.cast(size, tf.float32)
    else:
        raise ValueError("Either size or scale_factor must be defined.")
    
    if recompute_scale_factor:
        if scale_factor is None:
            raise ValueError("scale_factor must be defined if recompute_scale_factor is True.")
        # Recompute the scale factor based on the new size
        scale_factor_height = new_size[0] / tf.cast(input_shape[1], tf.float32)
        scale_factor_width = new_size[1] / tf.cast(input_shape[2], tf.float32)
        new_size = tf.stack([tf.cast(tf.cast(input_shape[1], tf.float32) * scale_factor_height, tf.float32),
                             tf.cast(tf.cast(input_shape[2], tf.float32) * scale_factor_width, tf.float32)])

    new_size = tf.cast(new_size, tf.int32)
    
    # Perform the interpolation
    if mode == 'bilinear':
        tf.compat.v1.image.resize_bilinear(input, size=new_size, align_corners=align_corners)
    elif mode == 'nearest':
        tf.compat.v1.image.resize_nearest_neighbor(input, size=new_size, align_corners=align_corners)
    elif mode == 'bicubic':
        tf.compat.v1.image.resize_bicubic(input, size=new_size, align_corners=align_corners)
    else:
        resize_result = tf.image.resize(input, size=new_size, method=mode, antialias=antialias)
    
    return resize_result
