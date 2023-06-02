import tensorflow as tf


def masking(inputs,mask_value=0.0,mask_mode="zero"):
    inputs=tf.convert_to_tensor(inputs)
    dtype=inputs.dtype
    mask=tf.equal(inputs,mask_value)
    mask=tf.cast(mask,dtype)
    mask=tf.broadcast_to(mask,tf.shape(inputs))
    if mask_mode=="zero":
        extreme_value=0
    elif mask_mode=="min":
        if dtype==tf.float32:
            extreme_value=tf.float32.min
        elif dtype==tf.float64:
            extreme_value=tf.float64.min
        elif dtype==tf.int32:
            extreme_value=tf.int32.min
        elif dtype==tf.int64:
            extreme_value=tf.int64.min
        else:
            raise ValueError("Unsupported dtype: {}".format(dtype))
    elif mask_mode=="max":
        if dtype==tf.float32:
            extreme_value=tf.float32.max
        elif dtype==tf.float64:
            extreme_value=tf.float64.max
        elif dtype==tf.int32:
            extreme_value=tf.int32.max
        elif dtype==tf.int64:
            extreme_value=tf.int64.max
        else:
            raise ValueError("Unsupported dtype: {}".format(dtype))
    else:
        raise ValueError("Invalid mask mode: {}".format(mask_mode))
    outputs=inputs*(1-mask)+mask*extreme_value
    return outputs,mask
