import tensorflow as tf
from Note.nn.layer.dense import dense
import dataclasses
from typing import Tuple

@dataclasses.dataclass
class RouterMask:
  """Dispatch and combine arrays for expert routing with masked matmuls.

  Attributes:
    dispatch_mask:
      <float>[num_groups, tokens_per_group, num_experts, expert_capacity]
      dispatch array that is 1 if the token gets routed to the
      corresponding expert, and 0 otherwise.
    combine_array:
      <float>[num_groups, tokens_per_group, num_experts, expert_capacity]
      combine array used for combining expert outputs and
      scaling with router probability.
  """
  dispatch_mask: tf.Tensor
  combine_array: tf.Tensor

RouterOutput = RouterMask


class router:
  """Abstract base router class, defining router API and inner workings.

  Computations are performed in float32 for stability, and returned after
  conversion according to the precision policy. See the discussion of
  "selective precision" in https://arxiv.org/abs/2101.03961.

  Uses Keras add_loss() and add_metric() APIs.

  Attributes:
    num_experts: Number of experts, used to check consistency with
      FeedForwardExperts.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    router_weights: Dense layer that computes logits for all tokens, which are
      then used as expert or token weights.
  """

  def __init__(
      self,
      num_experts: int,
      input_size=None,
      jitter_noise: float = 0.0,
      use_bias: bool = True,
      kernel_initializer = 'Xavier',
      bias_initializer = 'zeros',
      router_z_loss_weight: float = 0.0,
      export_metrics: bool = True,
      ):
    """Init.

    Args:
      num_experts: Number of experts.
      jitter_noise: Amplitude of jitter noise applied to router logits.
      use_bias: Whether or not to use the bias term in computing the router
        weights.
      kernel_initializer: Kernel initializer for router weights.
      bias_initializer: Bias initializer for router weights.
      router_z_loss_weight: Weight for router_z_loss. Use non-zero values if
        running into training instability (esp. with dtype 'bfloat16' or lower).
      export_metrics: Whether to export metrics using Keras add_metric API.
    """

    self.num_experts = num_experts  # Used to check consistency with
                                    # FeedForwardExperts.
    self.jitter_noise = jitter_noise
    self.router_z_loss_weight = router_z_loss_weight
    self._export_metrics = export_metrics

    self.router_weights = dense(
        num_experts,
        input_size,
        use_bias=use_bias,
        weight_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        dtype=tf.float32)

  def __call__(self,
           inputs: tf.Tensor,
           expert_capacity: int,
           train_flag = True) -> RouterOutput:
    """Computes dispatch and combine arrays for routing to experts.

    Args:
      inputs: Inputs to send to experts of shape
        <float>[num_groups, tokens_per_group, hidden_dim].
      expert_capacity: Each group will send this many tokens to each expert.
      train_flag: If true, apply jitter noise during routing. If not provided
        taken from tf.keras.backend.

    Returns:
      Router indices or mask arrays (depending on router type).
    """

    # inputs shape <float>[num_groups, tokens_per_group, hidden_dim]
    router_probs, router_logits = self._compute_router_probabilities(
        inputs, apply_jitter=train_flag)
    # router_probs <float32>[num_groups, tokens_per_group, num_experts]
    # router_logits <float>[num_groups, tokens_per_group, num_experts]
    unscaled_router_z_loss = _router_z_loss(router_logits)
    router_z_loss = self.router_z_loss_weight * unscaled_router_z_loss
    self.add_loss(router_z_loss)
    if self._export_metrics:
      self.add_metric(unscaled_router_z_loss, name="unscaled_router_z_loss")
      self.add_metric(router_z_loss, name="router_z_loss")

    routing_instructions = self._compute_routing_instructions(
        router_probs, expert_capacity)
    return routing_instructions

  def _compute_router_probabilities(
      self, inputs: tf.Tensor,
      apply_jitter: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes router probabilities from input tokens.

    Args:
      inputs: Inputs from which router probabilities are computed, shape
        <float>[num_groups, tokens_per_group, hidden_dim].
      apply_jitter: If true, apply jitter noise.

    Returns:
      - <float32>[num_groups, tokens_per_group, num_experts] probabilities for
        each token and expert. Used for routing tokens to experts.
      - <float32>[num_groups, tokens_per_group, num_experts] raw router logits.
        Used for computing router z-loss.
    """
    if apply_jitter and self.jitter_noise > 0:
      inputs *= tf.random.uniform(
          tf.shape(inputs),
          minval=1.0 - self.jitter_noise,
          maxval=1.0 + self.jitter_noise,
          dtype=inputs.dtype)
    # inputs <float>, router_logits <float32>
    router_logits = self.router_weights(inputs)
    router_probs = tf.nn.softmax(router_logits, axis=-1)
    return router_probs, router_logits

  def _compute_routing_instructions(self, router_probs: tf.Tensor,
                                    expert_capacity: int) -> RouterOutput:
    """Computes instructions for routing inputs to experts."""
    raise NotImplementedError(
        "Router is an abstract class that should be subclassed.")


class MaskedRouter(router):
  """Abstract base router class for masked matmul dispatch routers.

  MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine
  array for sending and receiving (via masked matmuls) inputs and outputs to and
  from experts.

  Routing using masked matmuls is generally faster than scatter-based routing on
  TPUs.

  Uses Keras add_loss() and add_metric() APIs.
  """

  def _compute_routing_instructions(self, router_probs: tf.Tensor,
                                    expert_capacity: int) -> RouterMask:
    """Computes masks for the top-k experts per token.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Router mask arrays.
    """
    raise NotImplementedError(
        "MaskedRouter is an abstract class that should be subclassed.")


def _router_z_loss(router_logits: tf.Tensor) -> float:
  """Computes router z-loss.

   The router z-loss was introduced in Designing Effective Sparse Expert Models
   (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
   small in an effort to improve stability.

  Args:
    router_logits: <float32>[num_groups, tokens_per_group, num_experts] router
      logits.

  Returns:
    Scalar router z-loss <float32>.
  """
  num_groups = tf.shape(router_logits)[0]
  tokens_per_group = router_logits.shape[1]

  log_z = tf.math.reduce_logsumexp(router_logits, axis=-1)
  z_loss = log_z**2
  return tf.math.reduce_sum(z_loss) / tf.cast(
      num_groups * tokens_per_group, tf.float32)