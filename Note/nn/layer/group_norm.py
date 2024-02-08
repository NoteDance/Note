import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Module import Module


class group_norm:
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization.

    Args:
      groups: Integer, the number of groups for Group Normalization. Can be in
        the range [1, N] where N is the input dimension. The input dimension
        must be divisible by the number of groups. Defaults to `32`.
      axis: Integer or List/Tuple. The axis or axes to normalize across.
        Typically, this is the features axis/axes. The left-out axes are
        typically the batch axis/axes. `-1` is the last dimension in the
        input. Defaults to `-1`.
      epsilon: Small float added to variance to avoid dividing by zero. Defaults
        to 1e-3
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored. Defaults to `True`.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling will be done by the next layer.
        Defaults to `True`.
      beta_initializer: Initializer for the beta weight. Defaults to zeros.
      gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        default.  Input shape: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis) when using this
        layer as the first layer in a model.  Output shape: Same shape as input.

    Call arguments:
      inputs: Input tensor (of any rank).
      mask: The mask parameter is a tensor that indicates the weight for each
        position in the input tensor when computing the mean and variance.

    Reference: - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups=32,
        input_size=None,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        mask=None,
        dtype='float32'
    ):
        self.input_size=input_size
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.mask=mask
        self.dtype = dtype
        self.param=[]
        if input_size!=None:
            if self.scale:
                self.gamma = initializer(input_size,gamma_initializer,dtype)
                self.param.append(self.gamma)
            else:
                self.gamma = None
    
            if self.center:
                self.beta = initializer(input_size,beta_initializer,dtype)
                self.param.append(self.beta)
            else:
                self.beta = None
            Module.param.extend(self.param)
    
    def build(self):
        if self.scale:
            self.gamma = initializer(self.input_size,self.gamma_initializer,self.dtype)
            self.param.append(self.gamma)
        else:
            self.gamma = None

        if self.center:
            self.beta = initializer(self.input_size,self.beta_initializer,self.dtype)
            self.param.append(self.beta)
        else:
            self.beta = None
        Module.param.extend(self.param)
        return

    def __call__(self, data):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        
        input_shape = tf.shape(data)

        if self.mask is None:
            mask = tf.ones_like(data)
        else:
            # We broadcast before we group in case the mask does not have the
            # same shape as the input.
            mask = tf.broadcast_to(self.mask, input_shape)

        reshaped_inputs = self._reshape_into_groups(data)
        reshaped_mask = self._reshape_into_groups(mask)

        normalized_inputs = self._apply_normalization(
            reshaped_inputs=reshaped_inputs,
            input_shape=input_shape,
            reshaped_mask=reshaped_mask,
        )

        return tf.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = [input_shape[i] for i in range(inputs.shape.rank)]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(
        self,
        *,
        reshaped_inputs,
        reshaped_mask,
        input_shape,
    ):
        group_reduction_axes = list(range(1, reshaped_inputs.shape.rank))

        axis = self.axis - 1
        group_reduction_axes.pop(axis)

        mask_weights = tf.cast(reshaped_mask, reshaped_inputs.dtype)

        mean, variance = tf.nn.weighted_moments(
            reshaped_inputs,
            axes=group_reduction_axes,
            frequency_weights=mask_weights,
            keepdims=True,
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * int(input_shape[0])

        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)

        return broadcast_shape