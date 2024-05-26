from random import randrange

import tensorflow as tf
from Note import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.tensorflow import Rearrange

# helpers

def exists(val):
    return val is not None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return tf.math.l2_normalize(t, axis=-1)

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = tf.random.uniform(shape=[num_layers], minval=0., maxval=1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale:
    def __init__(self, dim, fn, depth):
        if depth <= 18:
            init_eps = 0.1
        elif 18 > depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.fn = fn
        self.scale = tf.Variable(tf.fill((dim,), init_eps))

    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward:
    def __init__(self, dim, hidden_dim, dropout_rate = 0.):
        self.net = nn.Layers()
        self.net.add(nn.layer_norm(dim))
        self.net.add(nn.dense(hidden_dim, dim))
        self.net.add(tf.nn.gelu)
        self.net.add(nn.dropout(dropout_rate))
        self.net.add(nn.dense(dim, hidden_dim))
        self.net.add(nn.dropout(dropout_rate))
        
    def __call__(self, x, training):
        return self.net(x, training)

class Attention:
    def __init__(self, dim, heads = 8, dim_head = 64, dropout_rate = 0.):
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.layer_norm(dim)
        self.to_q = nn.dense(inner_dim, dim, use_bias = False)
        self.to_kv = nn.dense(inner_dim * 2, dim, use_bias = False)

        self.attend = tf.nn.softmax
        self.dropout = nn.dropout(dropout_rate)

        self.to_out = nn.Layers()
        self.to_out.add(nn.dense(dim, inner_dim))
        self.to_out.add(nn.dropout(dropout_rate))

    def __call__(self, x, training, context = None):
        h = self.heads

        x = self.norm(x)
        context = x if not exists(context) else tf.concat((x, context), axis = 1)

        qkv = (self.to_q(x), *tf.split(self.to_kv(context), 2, axis = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(sim)
        attn = self.dropout(attn, training)

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, training)

class XCAttention:
    def __init__(self, dim, heads = 8, dim_head = 64, dropout_rate = 0.):
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.layer_norm(dim)

        self.to_qkv = nn.dense(inner_dim * 3, dim, use_bias = False)

        self.temperature = nn.initializer_((heads, 1, 1), 'ones')

        self.attend = tf.nn.softmax
        self.dropout = nn.dropout(dropout_rate)

        self.to_out = nn.Layers()
        self.to_out.add(nn.dense(dim, inner_dim))
        self.to_out.add(nn.dropout(dropout_rate))

    def __call__(self, x, training):
        h = self.heads
        x, ps = pack_one(x, 'b * d')

        x = self.norm(x)
        q, k, v = tf.split(self.to_qkv(x), 3, axis = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = tf.einsum('b h i n, b h j n -> b h i j', q, k) * tf.exp(self.temperature)

        attn = self.attend(sim)
        attn = self.dropout(attn, training)

        out = tf.einsum('b h i j, b h j n -> b h i n', attn, v)
        out = rearrange(out, 'b h d n -> b n (h d)')

        out = unpack_one(out, ps, 'b * d')
        return self.to_out(out, training)

class LocalPatchInteraction:
    def __init__(self, dim, kernel_size = 3):
        assert (kernel_size % 2) == 1
        padding = kernel_size // 2

        self.net = nn.Layers()
        self.net.add(nn.layer_norm(dim))
        self.net.add(nn.zeropadding2d(padding = padding))
        self.net.add(nn.group_conv2d(dim, kernel_size, dim, dim))
        self.net.add(nn.batch_norm(dim))
        self.net.add(tf.nn.gelu)
        self.net.add(nn.zeropadding2d(padding = padding))
        self.net.add(nn.group_conv2d(dim, kernel_size, dim, dim))

    def __call__(self, x, training):
        return self.net(x, training)

class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        self.layers = []
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout_rate = dropout), depth = layer)
            ])

    def __call__(self, x, training, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, training = training, context = context) + x
            x = ff(x, training = training) + x

        return x

class XCATransformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size = 3, dropout = 0., layer_dropout = 0.):
        self.layers = []
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append([
                LayerScale(dim, XCAttention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout), depth = layer),
                LayerScale(dim, LocalPatchInteraction(dim, local_patch_kernel_size), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout_rate = dropout), depth = layer)
            ])

    def __call__(self, x, training):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for cross_covariance_attn, local_patch_interaction, ff in layers:
            x = cross_covariance_attn(x, training = training) + x
            x = local_patch_interaction(x, training = training) + x
            x = ff(x, training = training) + x

        return x

class XCiT(nn.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout_rate = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.dim = dim

        self.to_patch_embedding = nn.Layers()
        self.to_patch_embedding.add(Rearrange('b (h p1) (w p2) c -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size))
        self.to_patch_embedding.add(nn.layer_norm(patch_dim))
        self.to_patch_embedding.add(nn.dense(dim, patch_dim))
        self.to_patch_embedding.add(nn.layer_norm(dim))

        self.pos_embedding = nn.initializer_((1, num_patches, dim), 'normal')
        self.cls_token = nn.initializer_([dim], 'normal')

        self.dropout = nn.dropout(emb_dropout)

        self.xcit_transformer = XCATransformer(dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size, dropout_rate, layer_dropout)

        self.final_norm = nn.layer_norm(dim)

        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout_rate, layer_dropout)

        self.mlp_head = nn.Layers()
        self.mlp_head.add(nn.layer_norm(dim))
        self.mlp_head.add(nn.dense(num_classes, dim))
        
        self.training = True
        
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        self.flag=flag
        if flag==0:
            self.param_=self.param.copy()
            self.mlp_head_=self.mlp_head.layer[-1]
            self.mlp_head.layer[-1]=nn.dense(classes, self.dim)
            param.extend(self.mlp_head.layer[-1].param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.mlp_head.layer[-1].param):]
            self.param_.extend(self.mlp_head.layer[-1].param)
            self.param=self.param_
        else:
            self.mlp_head.layer[-1],self.mlp_head_=self.mlp_head_,self.mlp_head.layer[-1]
            del self.param_[-len(self.mlp_head.layer[-1].param):]
            self.param_.extend(self.mlp_head.layer[-1].param)
            self.param=self.param_
        return

    def __call__(self, img):
        x = self.to_patch_embedding(img)

        x, ps = pack_one(x, 'b * d')

        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        x = unpack_one(x, ps, 'b * d')

        x = self.dropout(x, self.training)

        x = self.xcit_transformer(x, self.training)

        x = self.final_norm(x)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)

        x = rearrange(x, 'b ... d -> b (...) d')
        cls_tokens = self.cls_transformer(cls_tokens, self.training, context = x)

        return self.mlp_head(cls_tokens[:, 0])