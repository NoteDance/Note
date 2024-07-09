import tensorflow as tf
from Note import nn

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = tf.constant([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)], dtype=tf.float32)
    pe[:, 0::2] = tf.math.sin(pe[:, 0::2])
    pe[:, 1::2] = tf.math.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# modules

class Attention:
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.dense(dim * 3, dim, use_bias=False)
        self.attn_drop = nn.dropout(attention_dropout)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(projection_dropout)

    def __call__(self, x):
        B, N, C = x.shape

        qkv = tf.split(self.qkv(x), 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = tf.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = tf.nn.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer:
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout_rate=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        self.pre_norm = nn.layer_norm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout_rate)

        self.linear1  = nn.dense(dim_feedforward, d_model)
        self.dropout1 = nn.dropout(dropout_rate)
        self.norm1    = nn.layer_norm(d_model)
        self.linear2  = nn.dense(d_model, dim_feedforward)
        self.dropout2 = nn.dropout(dropout_rate)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = tf.nn.gelu

    def __call__(self, src, training, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)), training)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src)), training))
        src = src + self.drop_path(self.dropout2(src2, training), training)
        return src

class DropPath:
    def __init__(self, drop_prob=None):
        self.drop_prob = float(drop_prob)

    def __call__(self, x, training):
        batch, drop_prob, dtype = x.shape[0], self.drop_prob, x.dtype

        if drop_prob <= 0. or not training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = tf.random.uniform(shape=shape, minval=0., maxval=1.) < keep_prob
        output = tf.math.divide(x, keep_prob) * tf.cast(keep_mask, 'float32')
        return output

class Tokenizer:
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        nn.Model.add()
        
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential()
        for chan_in, chan_out in n_filter_list_pairs:
            conv_layers = nn.Sequential()
            conv_layers.add(nn.conv2d(chan_out, (kernel_size, kernel_size), chan_in, strides=(stride, stride),
                                   padding=(padding, padding),use_bias=conv_bias))
            if not exists(activation):
                conv_layers.add(nn.identity())
            else:
                conv_layers.add(activation)
            if max_pool:
                conv_layers.add(nn.zeropadding2d(padding=pooling_padding))
                conv_layers.add(nn.max_pool2d(pooling_kernel_size, pooling_stride, 'VALID'))
            else:
                conv_layers.add(nn.identity())
            self.conv_layers.add(conv_layers)

        nn.Model.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.__call__(tf.zeros((1, height, width, n_channels))).shape[1]

    def __call__(self, x):
        return rearrange(self.conv_layers(x), 'b h w c -> b (h w) c')

    def init_weight(self, l):
        if isinstance(l, nn.conv2d):
            l.weight.assign(nn.initializer_(l.weight.shape, 'He'))


class TransformerClassifier:
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        nn.Model.add()
        
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.initializer((1, 1, self.embedding_dim), 'zeros')
        else:
            self.attention_pool = nn.dense(1, self.embedding_dim)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.initializer((1, sequence_length, embedding_dim), ['truncated_normal', 0.2])
        else:
            self.positional_emb = sinusoidal_embedding(sequence_length, embedding_dim)

        self.dropout = nn.dropout(dropout_rate)

        dpr = tf.linspace(0.0, stochastic_depth_rate, num_layers)

        self.blocks = []
        for layer_dpr in dpr:
            self.blocks.append(
                TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                        dim_feedforward=dim_feedforward, dropout_rate=dropout_rate,
                                        attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
                )

        self.norm = nn.layer_norm(embedding_dim)

        self.fc = nn.dense(num_classes, embedding_dim)
        nn.Model.apply(self.init_weight)

    def __call__(self, x, training):
        b = x.shape[0]

        if not exists(self.positional_emb) and x.shape[1] < self.sequence_length:
            x = tf.pad(x, (0, 0, 0, self.n_channels - x.shape[1]), mode='CONSTANT', constant_values=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = tf.concat((cls_token, x), axis=1)

        if exists(self.positional_emb):
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, training)

        x = self.norm(x)

        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = tf.einsum('b n, b n d -> b d', tf.nn.softmax(attn_weights), x)
        else:
            x = x[:, 0]

        return self.fc(x)

    def init_weight(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.trunc_normal_(l.weight, std=0.2))
            if l.use_bias is not False:
                l.bias.assign(nn.trunc_normal_(l.bias, std=0.2))
        

# CCT Main model

class CCT(nn.Model):
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args, **kwargs
    ):
        super().__init__()
        
        img_height, img_width = pair(img_size)
        self.embedding_dim = embedding_dim

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=tf.nn.relu,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)
        
        self.training = True

    def __call__(self, x):
        x = self.tokenizer(x)
        return self.classifier(x, self.training)