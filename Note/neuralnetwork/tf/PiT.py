import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.dropout import dropout
from Note.nn.layer.unfold import unfold
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.identity import identity
from Note.nn.initializer import initializer_
from Note.nn.Layers import Layers
from Note.nn.Module import Module
from math import sqrt
from einops import rearrange, repeat


def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


class FeedForward:
    def __init__(self, dim, hidden_dim, dropout_rate = 0.):
        self.net = Layers()
        self.net.add(layer_norm(dim))
        self.net.add(dense(hidden_dim, dim))
        self.net.add(tf.nn.gelu)
        self.net.add(dropout(dropout_rate))
        self.net.add(dense(dim, hidden_dim))
        self.net.add(dropout(dropout_rate))
        
    def __call__(self, x, training):
        return self.net(x, training)

class Attention:
    def __init__(self, dim, heads = 8, dim_head = 64, dropout_rate = 0.):
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = layer_norm(dim)
        self.attend = tf.nn.softmax
        self.dropout = dropout(dropout_rate)
        self.to_qkv = dense(inner_dim * 3, dim, use_bias = False)

        self.to_out = Layers()
        if project_out:
            self.to_out.add(dense(dim, inner_dim))
            self.to_out.add(dropout(dropout_rate))
        else:
            self.to_out.add(identity())

    def __call__(self, x, training):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        qkv = tf.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn, training)

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, training)

class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                Attention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout),
                FeedForward(dim, mlp_dim, dropout_rate = dropout)
            ])
        self.train_flag=True
            
    def __call__(self, x, training):
        for attn, ff in self.layers:
            x = attn(x, training) + x
            x = ff(x, training) + x
        return x

# pooling layer

class Pool:
    def __init__(self, dim):
        self.zeropadding2d = zeropadding2d(padding=1)
        self.downsample = depthwise_conv2d(input_size = dim, depth_multiplier = 2, kernel_size = 3, strides = 2)
        self.cls_ff = dense(dim * 2, dim)

    def __call__(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b h w c', h = int(sqrt(tokens.shape[1])))
        tokens = self.zeropadding2d(tokens)
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b h w c -> b (h w) c')

        return tf.concat((cls_token, tokens), axis = 1)


class PiT:
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout_rate = 0.,
        emb_dropout = 0.,
        channels = 3
    ):
        Module.init()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size ** 2
        self.dim = dim

        self.to_patch_embedding = Layers()
        self.to_patch_embedding.add(unfold(kernel = patch_size, stride = patch_size // 2))
        self.to_patch_embedding.add(dense(dim, patch_dim))

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2

        self.pos_embedding = initializer_((1, num_patches + 1, dim), 'normal')
        self.cls_token = initializer_((1, 1, dim), 'normal')
        self.dropout = dropout(emb_dropout)

        layers = Layers()

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            layers.add(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout_rate))

            if not_last:
                layers.add(Pool(dim))
                dim *= 2

        self.layers = layers

        self.mlp_head = Layers()
        self.mlp_head.add(layer_norm(dim))
        self.mlp_head.add(dense(num_classes, dim))
        
        self.param = Module.param
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
        x = self.to_patch_embedding(data)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = tf.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x, self.training)

        x = self.layers(x, self.training)

        return self.mlp_head(x[:, 0])