import tensorflow as tf
import math
import warnings


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + tf.math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in the truncated normal initialization. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1]
    tensor.assign(tf.random.uniform(tensor.shape, minval=2 * l - 1, maxval=2 * u - 1))

    # Use inverse cdf transform for normal distribution to get truncated standard normal
    tensor.assign(tf.math.erfinv(tensor))

    # Transform to proper mean, std
    tensor.assign(tensor * (std * math.sqrt(2.)) + mean)

    # Clamp to ensure it's in the proper range
    tensor.assign(tf.clip_by_value(tensor, clip_value_min=a, clip_value_max=b))

    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `tf.Variable`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> tensor = tf.Variable(tf.zeros((3, 5)), dtype=tf.float32)
        >>> trunc_normal_(tensor)
    """
    return _trunc_normal_(tensor, mean, std, a, b)


def trunc_normal_tf_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `tf.Variable`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> tensor = tf.Variable(tf.zeros((3, 5)), dtype=tf.float32)
        >>> trunc_normal_tf_(tensor)
    """
    _trunc_normal_(tensor, 0, 1.0, a, b)
    tensor.assign(tensor * std + mean)
    return tensor
