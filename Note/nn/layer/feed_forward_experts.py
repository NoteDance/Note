import tensorflow as tf
from Note.nn.layer.einsum_dense import einsum_dense
from Note.nn.layer.dropout import dropout

class feed_forward_experts:
  """Feed-forward layer with multiple experts.

  Note that call() takes inputs with shape
  [num_groups, num_experts, expert_capacity, hidden_dim]
  which is different from the usual [batch_size, seq_len, hidden_dim] used by
  the FeedForward layer.

  The experts are independent FeedForward layers of the
  same shape, i.e. the kernel doesn't have shape [hidden_dim, out_dim], but
  [num_experts, hidden_dim, out_dim].
  """

  def __init__(
      self,
      num_experts: int,
      d_ff: int,
      input_shape=None,
      inner_dropout: float = 0.0,
      output_dropout: float = 0.0,
      activation = tf.nn.gelu,
      kernel_initializer = 'Xavier',
      bias_initializer = 'zeros',
      ):
    """Initializes layer.

    Args:
      num_experts: Number of experts (i.e. number of independent feed-forward
        blocks).
      d_ff: Dimension of feed-forward layer of each expert.
      inner_dropout: The dropout probability to be applied after intermediate
        activations.
      output_dropout: The dropout probability to be applied after output layer.
      activation: (Nonlinear) transform applied in layer.
      kernel_initializer: Initialization scheme for kernel.
      bias_initializer: Initialization scheme for bias.
    """
    self.num_experts = num_experts
    self.d_ff = d_ff
    self.input_shape = input_shape
    self.inner_dropout = inner_dropout
    self.output_dropout = output_dropout
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    if input_shape is not None:
        self.intermediate_layer = einsum_dense(
            "gech,ehf->gecf",
            output_shape=(self.num_experts, None, d_ff),
            input_shape=input_shape,
            bias_axes="ef",
            weight_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            )
        self.inner_dropout_layer = dropout(
            inner_dropout)
        self.output_dropout_layer = dropout(output_dropout)
    
        """Creates the input shape dependent output weight variables."""
        if input_shape[1] != self.num_experts:
          raise ValueError(
              f"Input shape {input_shape} is inconsistent with num_experts "
              f"{self.num_experts}.")
    
        self.output_layer = einsum_dense(
            "gecf,efh->gech",
            output_shape=(self.num_experts, None, input_shape[-1]),
            input_shape=input_shape,
            bias_axes="eh",
            weight_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            )

  def build(self):
    self.intermediate_layer = einsum_dense(
        "gech,ehf->gecf",
        output_shape=(self.num_experts, None, self.d_ff),
        input_shape=self.input_shape,
        bias_axes="ef",
        weight_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        )
    self.inner_dropout_layer = dropout(
        self.inner_dropout)
    self.output_dropout_layer = dropout(self.output_dropout)

    """Creates the input shape dependent output weight variables."""
    if self.input_shape[1] != self.num_experts:
      raise ValueError(
          f"Input shape {self.input_shape} is inconsistent with num_experts "
          f"{self.num_experts}.")

    self.output_layer = einsum_dense(
        "gecf,efh->gech",
        output_shape=(self.num_experts, None, self.input_shape[-1]),
        input_shape=self.input_shape,
        bias_axes="eh",
        weight_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        )
    return

  def output(self,
           data: tf.Tensor,
           train_flag = True):
    """Applies layer to inputs.

    Args:
      inputs: Inputs of shape
        <float>[num_groups, num_experts, expert_capacity, hidden_dim].
      train_flag: Only apply dropout during training.

    Returns:
      Transformed inputs with the same shape as inputs
        <float>[num_groups, num_experts, expert_capacity, hidden_dim].
    """
    if self.input_shape==None:
        self.input_shape=data.shape
        self.build()
    x = self.intermediate_layer.output(data)
    x = self.activation(x)
    x = self.inner_dropout_layer.output(x, train_flag=train_flag)
    x = self.output_layer.output(x)
    x = self.output_dropout_layer.output(x, train_flag=train_flag)
    return x