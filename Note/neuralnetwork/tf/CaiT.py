from random import randrange
import tensorflow as tf
from Note.nn.Layers import Layers
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.dropout import dropout
from Note.nn.initializer import initializer_
from Note.nn.Module import Module

from einops import rearrange, repeat


def exists(val):
    return val is not None

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


class LayerScale:
    def __init__(self, dim, fn, depth):
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = tf.fill([1, 1, dim], init_eps)
        self.scale = tf.Variable(scale)
        Module.param.append(self.scale)
        self.fn = fn
        
    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward:
    def __init__(self, dim, hidden_dim, dropout_rate = 0.):
        self.net = Layers()
        self.net.add(layer_norm(dim))
        self.net.add(dense(hidden_dim, dim))
        self.net.add(tf.nn.gelu)
        self.net.add(dropout(dropout_rate))
        self.net.add(dense(dim, hidden_dim))
        self.net.add(dropout(dropout_rate))
        
    def __call__(self, x, training=True):
        return self.net(x, training)

class Attention:
    def __init__(self, dim, heads = 8, dim_head = 64, dropout_rate = 0.):
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = layer_norm(dim)
        self.to_q = dense(inner_dim, dim, use_bias = False)
        self.to_kv = dense(inner_dim * 2, dim, use_bias = False)

        self.attend = tf.nn.softmax
        self.dropout = dropout(dropout_rate)

        self.mix_heads_pre_attn = initializer_((heads, heads), 'normal')
        self.mix_heads_post_attn = initializer_((heads, heads), 'normal')

        self.to_out = Layers()
        self.to_out.add(dense(dim, inner_dim))
        self.to_out.add(dropout(dropout_rate))

    def __call__(self, x, context = None, training=True):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = x if not exists(context) else tf.concat((x, context), axis = 1)

        qkv = (self.to_q(x), *tf.split(self.to_kv(context), 2, axis = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = tf.einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax

        attn = self.attend(dots)
        attn = self.dropout(attn, training)

        attn = tf.einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, training)

class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        self.layers = []
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout), depth = ind + 1),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout_rate = dropout), depth = ind + 1)
            ])
            
    def __call__(self, x, context = None, training=True):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context, training=training) + x
            x = ff(x, training=training) + x
        return x

class CaiT:
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
        layer_dropout = 0.
    ):
        Module.init()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.patch_size = patch_size

        self.to_patch_embedding = Layers()
        self.to_patch_embedding.add(layer_norm(patch_dim))
        self.to_patch_embedding.add(dense(dim, patch_dim))
        self.to_patch_embedding.add(layer_norm(dim))

        self.pos_embedding = initializer_((1, num_patches, dim), 'normal')
        self.cls_token = initializer_((1, 1, dim), 'normal')

        self.dropout = dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout_rate, layer_dropout)

        self.mlp_head = Layers()
        self.mlp_head.add(layer_norm(dim))
        self.mlp_head.add(dense(num_classes, dim))
        
        self.param=Module.param
        self.training=True
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.mlp_head_=self.mlp_head.layer[-1]
            self.mlp_head.layer[-1]=dense(classes, self.dim)
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

    def __call__(self, data):
        b = data.shape[0]
        h = data.shape[1] // self.patch_size
        w = data.shape[2] // self.patch_size
        c = data.shape[3]
        data = tf.reshape(data, (b, h * w, self.patch_size * self.patch_size * c))
        x = self.to_patch_embedding(data)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x, self.training)

        x = self.patch_transformer(x, training=self.training)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x, training=self.training)

        return self.mlp_head(x[:, 0])