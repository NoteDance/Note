import tensorflow as tf
from Note.nn.layer.feed_forward_experts import feed_forward_experts
from Note.nn.layer.router import MaskedRouter


class MoE_layer:
  """Sparse MoE layer with per-token routing.

  Attributes:
    num_experts: Number of experts (i.e. number of independent feed-forward
      blocks).
  """

  def __init__(
      self,
      experts: feed_forward_experts,
      router: MaskedRouter,
      train_capacity_factor: float = 1.0,
      eval_capacity_factor: float = 1.0,
      examples_per_group: float = 1.0,
      ):
    """Init.

    Args:
      experts: Instance of FeedForwardExperts. Needs to have the same
        num_experts as the router.
      router: Instance of MaskedRouter to route the tokens to
        the different experts.
      train_capacity_factor: Scaling factor to increase the expert token
        capacity during training. This factor plays an analogous, but slightly
        different, role depending on the routing assignment algorithm:
        - For "tokens choose" routing, the capacity factor only affects the
          maximum number of tokens that an expert will process. It does not
          affect how many experts a given token is routed to; see the
          num_selected_experts attributes of "tokens choose" routers.
        - For "experts choose" routing, because experts always fill their
          buffer, increasing the capacity factor will increase the number of
          tokens that an expert will process AND will indirectly increase the
          number of experts that a given token is routed to.
      eval_capacity_factor: As above, but used during evaluation.
      examples_per_group: Number of examples to form a group. Router then
        performs top_k token selection for each expert on a per group basis.
        E.g. when `examples_per_group=4.0`, tokens are assigned to experts in
        groups formed from 4 examples. When `examples_per_group=0.5`,
        each example is split into 2 groups.
        `examples_per_group` must divide the local batch size.
        A larger group size will result in slower but more accurate top-k and
        sorting computations, whereas a smaller group size will result in faster
        but more approximate (and potentially less stable) routing choices.
        In practice, we find that imperfect routing choices are tolerable and
        recommend choosing a group size on the order of 4096 tokens, although
        this number will vary based on model configuration and size.
    """
    self._experts = experts
    self._router = router

    self.num_experts = experts.num_experts
    assert experts.num_experts == router.num_experts

    self._train_capacity_factor = train_capacity_factor
    self._eval_capacity_factor = eval_capacity_factor
    self._examples_per_group = examples_per_group

  def __call__(self,
           inputs: tf.Tensor,
           train_flag = True) -> tf.Tensor:
    """Applies MoeLayer.

    Args:
      inputs: Batch of input embeddings of shape
        <float>[batch_size, seq_length, hidden_dim].
      training: Only apply dropout and jitter noise during training. If not
        provided taken from tf.keras.backend.

    Returns:
      Transformed inputs with same shape as inputs:
        <float>[batch_size, seq_length, hidden_dim].

    Raises:
      ValueError if we cannot find a group_size satisfying given requirements.
    """

    # inputs shape [batch_size, seq_length, hidden_dim]
    batch_size, seq_length, hidden_dim = inputs.shape
    if batch_size is not None:
      if self._examples_per_group > batch_size:
        raise ValueError(
            f"examples_per_group={self._examples_per_group} is larger than the "
            "number of examples available in the local (per-device) batch_size="
            f"{batch_size}. Either decrease examples_per_group or increase the "
            "batch_size.")
    tokens_per_group = int(seq_length * self._examples_per_group)

    if train_flag:
      capacity_factor = self._train_capacity_factor
    else:
      capacity_factor = self._eval_capacity_factor
    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(capacity_factor * tokens_per_group / self.num_experts))

    # Reshape batch and sequence/token dimensions for expert routing.
    x = tf.reshape(inputs, (-1, tokens_per_group, hidden_dim))

    x = self._mask_and_dispatch_to_experts(x, expert_capacity, train_flag)

    # Return to original input shape.
    x = tf.reshape(x, (-1, seq_length, hidden_dim))
    return x

  def _mask_and_dispatch_to_experts(self, inputs: tf.Tensor,
                                    expert_capacity: int,
                                    train_flag: bool) -> tf.Tensor:
    """Wraps expert masked routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute dispatch mask and combine array using self._router.
    (2) Dispatch inputs to experts based on dispatch mask.
    (3) Recombine individual expert outputs using combine array.

    Args:
      inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      expert_capacity: Each group will send this many tokens to each expert.
      training: If true, apply jitter noise during routing and dropout
        during expert computation.

    Returns:
      <float>[num_groups, num_tokens_per_group, hidden_dim] outputs from
        experts.
    """
    # Shape [num_groups, tokens_per_group, num_experts, expert_capacity]
    router_mask = self._router(
        inputs,
        expert_capacity=expert_capacity,
        training=train_flag)

    # Shape [num_groups, num_experts, expert_capacity, hidden_dim]
    expert_inputs = tf.einsum(
        "gtec,gth->gech",
        router_mask.dispatch_mask,
        inputs)

    expert_outputs = self._experts(expert_inputs, train_flag=train_flag)

    # Shape [num_groups, tokens_per_group, hidden_dim]
    combined_outputs = tf.einsum(
        "gtec,gech->gth",
        router_mask.combine_array,
        expert_outputs)

    return combined_outputs