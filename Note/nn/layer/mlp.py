import tensorflow as tf
from Note import nn
from functools import partial


class Mlp:
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.gelu,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = nn.to_2tuple(bias)
        drop_probs = nn.to_2tuple(drop)
        linear_layer = partial(nn.conv2d, kernel_size=1) if use_conv else nn.dense

        self.fc1 = linear_layer(hidden_features, input_size=in_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.identity()
        self.fc2 = linear_layer(out_features, input_size=hidden_features, use_bias=bias[1])
        self.drop2 = nn.dropout(drop_probs[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp:
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = nn.to_2tuple(bias)
        drop_probs = nn.to_2tuple(drop)
        linear_layer = partial(nn.conv2d, kernel_size=1) if use_conv else nn.dense
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(hidden_features, input_size=in_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.identity()
        self.fc2 = linear_layer(out_features, input_size=hidden_features // 2, use_bias=bias[1])
        self.drop2 = nn.dropout(drop_probs[1])

    def __call__(self, x):
        x = self.fc1(x)
        x1, x2 = tf.split(x, 2, axis=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=tf.nn.silu, gate_last=False)


class SwiGLU:
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.silu,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = nn.to_2tuple(bias)
        drop_probs = nn.to_2tuple(drop)

        self.fc1_g = nn.dense(hidden_features, in_features, use_bias=bias[0])
        self.fc1_x = nn.dense(hidden_features, in_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.identity()
        self.fc2 = nn.dense(out_features, hidden_features, use_bias=bias[1])
        self.drop2 = nn.dropout(drop_probs[1])

    def __call__(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
    
class GatedMlp:
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.gelu,
            norm_layer=None,
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = nn.to_2tuple(bias)
        drop_probs = nn.to_2tuple(drop)

        self.fc1 = nn.dense(hidden_features, in_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.identity()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.identity()
        self.fc2 = nn.dense(out_features, hidden_features, use_bias=bias[1])
        self.drop2 = nn.dropout(drop_probs[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp:
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.relu,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = nn.to_2tuple(bias)

        self.fc1 = nn.conv2d(hidden_features, 1, in_features, use_bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.identity()
        self.act = act_layer
        self.drop = nn.dropout(drop)
        self.fc2 = nn.conv2d(out_features, 1, hidden_features, use_bias=bias[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp:
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.gelu,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = nn.to_2tuple(bias)
        drop_probs = nn.to_2tuple(drop)
        linear_layer = partial(nn.conv2d, kernel_size=1) if use_conv else nn.dense

        self.fc1 = linear_layer(hidden_features, input_size=in_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=True)
        self.fc2 = linear_layer(out_features, input_size=hidden_features, use_bias=bias[1])
        self.drop2 = nn.dropout(drop_probs[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GlobalResponseNorm:
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = nn.variable(tf.zeros(dim))
        self.bias = nn.variable(tf.zeros(dim))

    def __call__(self, x):
        x_g = tf.norm(x, ord=2, axis=self.spatial_dim, keepdims=True)
        x_n = x_g / (tf.reduce_mean(x_g, axis=self.channel_dim, keepdims=True) + self.eps)
        x_mul_xn = tf.multiply(x, x_n)
        weighted_product = tf.multiply(x_mul_xn, tf.reshape(self.weight, self.wb_shape))
        return x + tf.add(tf.reshape(self.bias, self.wb_shape), weighted_product)
