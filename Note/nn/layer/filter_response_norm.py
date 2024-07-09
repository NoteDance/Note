import tensorflow as tf
from Note.nn.initializer import initializer


class filter_response_norm:
    """Filter response normalization layer.

    Filter Response Normalization (FRN), a normalization
    method that enables models trained with per-channel
    normalization to achieve high accuracy. It performs better than
    all other normalization techniques for small batches and is par
    with Batch Normalization for bigger batch sizes.

    Arguments
        axis: List of axes that should be normalized. This should represent the
              spatial dimensions.
        epsilon: Small positive float value added to variance to avoid dividing by zero.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        learned_epsilon: (bool) Whether to add another learnable
        epsilon parameter or not.

    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model. This layer, as of now,
        works on a 4-D tensor where the tensor should have the shape [N X H X W X C]

        TODO: Add support for NCHW data format and FC layers.

    Output shape
        Same shape as input.

    References
        - [Filter Response Normalization Layer: Eliminating Batch Dependence
        in the training of Deep Neural Networks]
        (https://arxiv.org/abs/1911.09737)
    """

    def __init__(
        self,
        input_shape=None,
        epsilon: float = 1e-6,
        axis: list = [1, 2],
        beta_initializer = "zeros",
        gamma_initializer = "ones",
        learned_epsilon: bool = False,
        dtype = 'float32'
    ):
        self.epsilon = epsilon
        self.axis = axis
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.use_eps_learned = learned_epsilon
        self.dtype=dtype

        if self.use_eps_learned:
            self.eps_learned = tf.constant(1e-4,dtype)
        self.input_shape=input_shape
        if input_shape is not None:
            if len(input_shape) != 4:
                raise ValueError(
                    """Only 4-D tensors (CNNs) are supported
            as of now."""
                )
            self._check_if_input_shape_is_none(input_shape)
            self._add_gamma_weight(input_shape)
            self._add_beta_weight(input_shape)
        
        self._check_axis(axis)
            

    def __call__(self, data):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_shape is None:
            self.input_shape=data.shape
            self._check_if_input_shape_is_none(self.input_shape)
            self._add_gamma_weight(self.input_shape)
            self._add_beta_weight(self.input_shape)
        epsilon = tf.math.abs(tf.cast(self.epsilon, dtype=self.dtype))
        if self.use_eps_learned:
            epsilon += tf.math.abs(self.eps_learned)
        nu2 = tf.reduce_mean(tf.square(data), axis=self.axis, keepdims=True)
        normalized_inputs = data * tf.math.rsqrt(nu2 + epsilon)
        return self.gamma * normalized_inputs + self.beta
    
    
    def _add_gamma_weight(self, input_shape):
        # Get the channel dimension
        dim = input_shape[-1]
        shape = [1, 1, 1, dim]
        # Initialize gamma with shape (1, 1, 1, C)
        self.gamma = initializer(shape, self.gamma_initializer, self.dtype)
        return


    def _add_beta_weight(self, input_shape):
        # Get the channel dimension
        dim = input_shape[-1]
        shape = [1, 1, 1, dim]
        # Initialize beta with shape (1, 1, 1, C)
        self.beta = initializer(shape, self.beta_initializer, self.dtype)
        return
    
    
    def _check_axis(self, axis):
        if not isinstance(axis, list):
            raise TypeError(
                """Expected a list of values but got {}.""".format(type(axis))
            )
        else:
            self.axis = axis
    
        if self.axis != [1, 2]:
            raise ValueError(
                """FilterResponseNormalization operates on per-channel basis.
                Axis values should be a list of spatial dimensions."""
            )
            

    def _check_if_input_shape_is_none(self, input_shape):
        dim1, dim2 = input_shape[self.axis[0]], input_shape[self.axis[1]]
        if dim1 is None or dim2 is None:
            raise ValueError(
                """Axis {} of input tensor should have a defined dimension but
                the layer received an input with shape {}.""".format(
                    self.axis, input_shape
                )
            )