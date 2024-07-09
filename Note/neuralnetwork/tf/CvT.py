import tensorflow as tf
from Note import nn

from einops import rearrange
from einops.layers.tensorflow import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class LayerNorm: # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        self.eps = eps
        self.g = nn.initializer((1, dim, 1, 1), 'ones')
        self.b = nn.initializer((1, dim, 1, 1), 'zeros')

    def __call__(self, x):
        var = tf.math.reduce_variance(x, axis=1, keepdims=True)
        mean = tf.reduce_mean(x, axis = 1, keepdims = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward:
    def __init__(self, dim, mult = 4, dropout_rate = 0.):
        self.net = nn.Sequential()
        self.net.add(nn.layer_norm(dim))
        self.net.add(nn.conv2d(dim * mult, 1, dim))
        self.net.add(tf.nn.gelu)
        self.net.add(nn.dropout(dropout_rate))
        self.net.add(nn.conv2d(dim, 1, dim * mult))
        self.net.add(nn.dropout(dropout_rate))

    def __call__(self, x, training):
        return self.net(x, training)

class DepthWiseConv2d:
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        self.net = nn.Sequential()
        self.net.add(nn.conv2d(dim_in, kernel_size, dim_in, groups=dim_in, strides = stride, padding=padding, use_bias = bias))
        self.net.add(nn.batch_norm(dim_in))
        self.net.add(nn.conv2d(dim_out, 1, dim_in, use_bias = bias))

    def __call__(self, x, training):
        return self.net(x, training)

class Attention:
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout_rate = 0.):
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.layer_norm(dim)
        self.attend = tf.nn.softmax
        self.dropout = nn.dropout(dropout_rate)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential()
        self.to_out.add(nn.conv2d(dim, 1, inner_dim))
        self.to_out.add(nn.dropout(dropout_rate))

    def __call__(self, x, training):
        shape = x.shape
        b, _, y, n, h = *shape, self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x, training), *tf.split(self.to_kv(x, training), 2, axis = -1))
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h = h), (q, k, v))

        dots = tf.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn, training)

        out = tf.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b x y (h d)', h = h, y = y)
        return self.to_out(out, training)

class Transformer:
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout_rate = dropout),
                FeedForward(dim, mlp_mult, dropout_rate = dropout)
            ])
        self.train_flag = True

    def __call__(self, x, training):
        for attn, ff in self.layers:
            x = attn(x, training) + x
            x = ff(x, training) + x
        return x

class CvT(nn.Model):
    def __init__(
        self,
        num_classes,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.,
        channels = 3
    ):
        super().__init__()

        kwargs = dict(locals())

        dim = channels
        self.dim = dim
        self.layers = nn.Sequential()

        for prefix in ('s1', 's2', 's3'):
            layers = nn.Sequential()
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.add(nn.conv2d(config['emb_dim'], config['emb_kernel'], dim, strides = config['emb_stride'], padding = (config['emb_kernel'] // 2)))
            layers.add(nn.layer_norm(config['emb_dim']))
            layers.add(Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout))
            self.layers.add(layers)

            dim = config['emb_dim']

        self.to_logits = nn.Sequential()
        self.to_logits.add(nn.adaptive_avg_pooling2d(1))
        self.to_logits.add(Rearrange('() () ... -> ...'))
        self.to_logits.add(nn.dense(num_classes, dim))

        self.training = True
        
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        self.flag=flag
        if flag==0:
            self.param_=self.param.copy()
            self.to_logits_=self.to_logits.layer[-1]
            self.to_logits.layer[-1]=nn.dense(classes, self.dim)
            param.extend(self.to_logits.layer[-1].param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.to_logits.layer[-1].param):]
            self.param_.extend(self.to_logits.layer[-1].param)
            self.param=self.param_
        else:
            self.to_logits.layer[-1],self.to_logits_=self.to_logits_,self.to_logits.layer[-1]
            del self.param_[-len(self.to_logits.layer[-1].param):]
            self.param_.extend(self.to_logits.layer[-1].param)
            self.param=self.param_
        return

    def __call__(self, data):
        latents = self.layers(data, self.training)
        return self.to_logits(latents)