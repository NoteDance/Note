import tensorflow as tf
from Note import nn


class PatchDropout:
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices
        self.train_flag=True
        nn.Model.layer_list.append(self)
        if nn.Model.name_!=None and nn.Model.name_ not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name_]=[]
            nn.Model.layer_eval[nn.Model.name_].append(self)
        elif nn.Model.name_!=None:
            nn.Model.layer_eval[nn.Model.name_].append(self)

    def __call__(self, x, training=None):
        if training==None:
            training=self.train_flag
        if not training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = tf.argsort(tf.random.normal((B, L)), axis=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = tf.sort(keep_indices, axis=-1)
        x = tf.gather(x, keep_indices, axis=1, batch_dims=1)

        if prefix_tokens is not None:
            x = tf.concat((prefix_tokens, x), axis=1)

        if self.return_indices:
            return x, keep_indices
        return x