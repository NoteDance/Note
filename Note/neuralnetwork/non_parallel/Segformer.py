import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.up_sampling2d import up_sampling2d
from Note.nn.initializer import initializer_
from Note.nn.Layers import Layers
from Note.nn.Module import Module
from einops import rearrange
from math import sqrt
from functools import partial

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d:
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        self.net = Layers()
        self.net.add(conv2d(dim_in, input_size = dim_in, kernel_size = kernel_size, strides = stride, use_bias = bias))
        self.net.add(zeropadding2d(dim_in, padding))
        self.net.add(conv2d(dim_out, input_size = dim_in, kernel_size = 1, use_bias = bias))
        
    def __call__(self, x):
        return self.net(x)

class LayerNorm:
    def __init__(self, dim, eps = 1e-5):
        self.eps = eps
        self.g = initializer_((1, dim, 1, 1), 'ones', 'float32')
        self.b = initializer_((1, dim, 1, 1), 'zeros', 'float32')

    def __call__(self, x):
        std = tf.math.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True))
        mean = tf.reduce_mean(x, axis= 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm:
    def __init__(self, dim, fn):
        self.fn = fn
        self.norm = layer_norm(dim)

    def __call__(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention:
    def __init__(
        self,
        dim,
        heads,
        reduction_ratio
    ):
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = conv2d(dim, 1, input_size = dim, use_bias = False)
        self.to_kv = conv2d(dim * 2, reduction_ratio, input_size = dim, strides = reduction_ratio, use_bias = False)
        self.to_out = conv2d(dim, 1, input_size = dim, use_bias = False)
        self.output_size = self.to_out.output_size

    def __call__(self, x):
        h, w = x.shape[1], x.shape[2]
        heads = self.heads

        q, k, v = (self.to_q(x), *tf.split(self.to_kv(x), num_or_size_splits=2, axis=-1))
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> (b h) (x y) c', h = heads), (q, k, v))
        
        sim = tf.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = tf.nn.softmax(sim)

        out = tf.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b x y (h c)', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward:
    def __init__(
        self,
        dim,
        expansion_factor
    ):
        hidden_dim = dim * expansion_factor
        self.net = Layers()
        self.net.add(conv2d(hidden_dim, 1, dim))
        self.net.add(DsConv2d(hidden_dim, hidden_dim, 3, padding = 1))
        self.net.add(tf.nn.gelu)
        self.net.add(conv2d(dim, 1, hidden_dim))
        self.output_size = self.net.output_size

    def __call__(self, x):
        return self.net(x)


class Unfold:
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.zeropadding2d = zeropadding2d(padding=padding)
    
    def __call__(self, x):
        x = self.zeropadding2d(x)
        x = tf.image.extract_patches(x, sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.stride, self.stride, 1], rates=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))
        return x


class MiT:
    def __init__(
        self,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = []
        
        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = Unfold(kernel, stride, padding)
            overlap_patch_embed = conv2d(dim_out, 1, dim_in * kernel ** 2)

            layers = []

            for _ in range(num_layers):
                layers.append([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ])

            self.stages.append([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ])

    def __call__(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[1], x.shape[2]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-2]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b (h w) c -> b h w c', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer:
    def __init__(
        self,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        Module.init()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        
        self.to_fused = []
        for i, dim in enumerate(dims):
            to_fused = Layers()
            to_fused.add(conv2d(decoder_dim, 1, dim))
            to_fused.add(up_sampling2d(2 ** i))
            self.to_fused.append(to_fused)

        self.to_segmentation = Layers()
        self.to_segmentation.add(conv2d(decoder_dim, 1, 4 * decoder_dim))
        self.to_segmentation.add(conv2d(num_classes, 1, decoder_dim))
        
        self.opt = tf.keras.optimizers.Adam()
        self.param = Module.param
    
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.conv2d=self.to_segmentation.layer[-1]
            self.to_segmentation.layer[-1]=conv2d(classes, input_size=self.conv2d.input_size, kernel_size=1)
            param.extend(self.to_segmentation.layer[-1].param)
            self.param=param
            self.opt.lr=lr
        elif flag==1:
            del self.param_[-len(self.to_segmentation.layer[-1].param):]
            self.param_.extend(self.to_segmentation.layer[-1].param)
            self.param=self.param_
        else:
            self.to_segmentation.layer[-1],self.conv2d=self.conv2d,self.to_segmentation.layer[-1]
            del self.param_[-len(self.to_segmentation.layer[-1].param):]
            self.param_.extend(self.to_segmentation.layer[-1].param)
            self.param=self.param_
        return

    def fp(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)
    
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = tf.concat(fused, axis = -1)
        return self.to_segmentation(fused)
    
    def loss(self, output, labels):
        output = tf.reshape(output, [-1, output.shape[-1]])
        labels = tf.reshape(labels, [-1, labels.shape[-1]])
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)