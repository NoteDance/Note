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


def dirac_(tensor, groups=1):
    r"""Fill the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function.

    Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (int, optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = tf.Variable(tf.zeros([3, 16, 5, 5]))
        >>> nn.dirac_(w)
        >>> w = tf.Variable(tf.zeros([3, 24, 5, 5]))
        >>> nn.dirac_(w, 3)
    """
    dimensions = tensor.shape.ndim
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.shape

    if sizes[-1] % groups != 0:
        raise ValueError('dim 0 must be divisible by groups')

    out_chans_per_grp = sizes[-1] // groups
    min_dim = min(out_chans_per_grp, sizes[-2])

    tensor.assign(tf.zero(tensor.shape))

    for g in range(groups):
        for d in range(min_dim):
            if dimensions == 3:  # Temporal convolution
                tensor[tensor.shape[0] // 2, d, g * out_chans_per_grp + d].assign(1)
            elif dimensions == 4:  # Spatial convolution
                tensor[tensor.shape[0] // 2, tensor.shape[1] // 2, 
                           d, g * out_chans_per_grp + d].assign(1)
            else:  # Volumetric convolution
                tensor[tensor.shape[0] // 2, tensor.shape[1] // 2, tensor.shape[2] // 2, 
                           d, g * out_chans_per_grp + d].assign(1)
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.shape.ndims
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.shape.ndims > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.assign(tf.random.normal(tensor.shape, stddev=math.sqrt(variance)))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.assign(tf.random.uniform(tensor.shape, -bound, bound))
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def xavier_uniform_(
    tensor, gain: float = 1.0, generator = None
):
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = tf.Variable(tf.zeros([3, 5]))
        >>> nn.xavier_uniform_(w, gain=nn.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    if generator==None:
        return tensor.assign(tf.random.uniform(tensor, -a, a))
    else:
        return tensor.assign(generator.uniform(tensor, -a, a))


def xavier_normal_(
    tensor,
    gain: float = 1.0,
    generator = None,
):
    r"""Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = tf.Variable(tf.zeros([3, 5]))
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    if generator==None:
        return tensor.assign(tf.random.normal(tensor.shape, 0., std))
    else:
        return tensor.assign(generator.normal(tensor.shape, 0., std))


def kaiming_uniform_(
    tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator = None,
):
    r"""Fill the input `Tensor` with values using a Kaiming uniform distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)
        
    Examples:
        >>> w = tf.Variable(tf.zeros([3, 5]))
        >>> nn.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    if generator==None:
        return tensor.assign(tf.random.uniform(tensor.shape, -bound, bound))
    else:
        return tensor.assign(generator.uniform(tensor.shape, -bound, bound))


def kaiming_normal_(
    tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator = None,
):
    r"""Fill the input `Tensor` with values using a Kaiming normal distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)
        
    Examples:
        >>> w = tf.Variable(tf.zeros([3, 5]))
        >>> nn.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    if generator==None:
        return tensor.assign(tf.random.normal(tensor.shape, 0, std))
    else:
        return tensor.assign(generator.normal(tensor.shape, 0, std))
    
    
def orthogonal_(
    tensor,
    gain=1,
    generator = None,
):
    r"""Fill the input `Tensor` with a (semi) orthogonal matrix.

    Described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> w = tf.Variable(tf.zeros([3, 5]))
        >>> nn.orthogonal_(w)
    """
    if tensor.shape.ndims < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if tensor.numel() == 0:
        # no-op
        return tensor
    rows = tensor.shape[0]
    cols = tf.size(tensor).numpy() // rows
    if generator==None:
        flattened = tf.Variable(tf.random.normal((rows, cols), 0, 1))
    else:
        flattened = tf.Variable(generator.normal((rows, cols), 0, 1))

    if rows < cols:
        tf.transpose(flattened)

    # Compute the qr factorization
    q, r = tf.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = tf.linalg.diag(r, 0)
    ph = tf.sign(d)
    q *= ph

    if rows < cols:
        tf.transpose(q)

    tensor.assign(tf.reshape(q, tensor.shape))
    tensor.assign(tensor * gain)
    return tf.reshape(tensor, q.shape)


def sparse_(
    tensor,
    sparsity,
    std=0.01,
    generator = None,
):
    r"""Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = tf.Variable(tf.zeros([3, 5]))
        >>> nn.sparse_(w, sparsity=0.1)
    """
    if tensor.shape.ndims != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    if generator==None:
        tensor.assign(tf.random.normal(tensor.shape, 0, std))
    else:
        tensor.assign(generator.normal(tensor.shape, 0, std))
    for col_idx in range(cols):
        row_indices = tf.random.shuffle(tf.range(rows))
        zero_indices = row_indices[:num_zeros]
        for zero_idx in zero_indices:
            tensor[zero_idx, col_idx].assign(0)
    return tensor
