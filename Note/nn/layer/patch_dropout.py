import tensorflow as tf
from Note import nn
from typing import Optional, Tuple, Union


def patch_dropout_forward(
        x: tf.Tensor,
        prob: float,
        num_prefix_tokens: int,
        ordered: bool,
        training: bool,
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """
    Common forward logic for patch dropout.

    Args:
        x: Input tensor of shape (B, L, D)
        prob: Dropout probability
        num_prefix_tokens: Number of prefix tokens to preserve
        ordered: Whether to maintain patch order
        training: Whether in training mode

    Returns:
        Tuple of (output tensor, keep_indices or None)
    """
    if not training or prob == 0.:
        return x, None

    if num_prefix_tokens:
        prefix_tokens, x = x[:, :num_prefix_tokens], x[:, num_prefix_tokens:]
    else:
        prefix_tokens = None

    B = x.shape[0]
    L = x.shape[1]
    num_keep = max(1, int(L * (1. - prob)))
    keep_indices = tf.argsort(tf.random.normal([B, L]), axis=-1)[:, :num_keep]

    if ordered:
        # NOTE does not need to maintain patch order in typical transformer use,
        # but possibly useful for debug / visualization
        keep_indices = tf.sort(keep_indices, axis=-1)

    x = tf.gather(x, keep_indices, axis=1, batch_dims=1)

    if prefix_tokens is not None:
        x = tf.concat((prefix_tokens, x), axis=1)

    return x, keep_indices


class PatchDropout(nn.Layer):
    """
    Patch Dropout without returning indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        nn.Model.register(self)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        output, _ = patch_dropout_forward(
            x,
            self.prob,
            self.num_prefix_tokens,
            self.ordered,
            self.training
        )
        return output


class PatchDropoutWithIndices(nn.Layer):
    """
    Patch Dropout that returns both output and keep indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        nn.Model.register(self)

    def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return patch_dropout_forward(
            x,
            self.prob,
            self.num_prefix_tokens,
            self.ordered,
            self.training
        )
