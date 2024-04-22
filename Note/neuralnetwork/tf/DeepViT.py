import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.dropout import dropout
from Note.nn.layer.identity import identity
from Note.nn.initializer import initializer_
from Note.nn.Layers import Layers
from Note.nn.Module import Module

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

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
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = layer_norm(dim)
        self.to_qkv = dense(inner_dim * 3, dim, use_bias = False)

        self.dropout = dropout(dropout_rate)

        self.reattn_weights = initializer_((heads, heads), 'normal')

        self.reattn_norm = Layers()
        self.reattn_norm.add(Rearrange('b h i j -> b i j h'))
        self.reattn_norm.add(layer_norm(heads))
        self.reattn_norm.add(Rearrange('b i j h -> b h i j'))

        self.to_out = Layers()
        self.to_out.add(dense(dim, inner_dim))
        self.to_out.add(dropout(dropout_rate))

    def __call__(self, x, training):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)

        qkv = tf.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = tf.nn.softmax(dots)
        attn = self.dropout(attn, training)

        # re-attention

        attn = tf.einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out, training)
        return out

class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout_rate = 0.):
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                Attention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout_rate),
                FeedForward(dim, mlp_dim, dropout_rate = dropout_rate)
            ])
            
    def __call__(self, x, training):
        for attn, ff in self.layers:
            x = attn(x, training) + x
            x = ff(x, training) + x
        return x

class DeepViT:
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout_rate = 0., emb_dropout = 0.):
        Module.init()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.dim = dim

        self.to_patch_embedding = Layers()
        self.to_patch_embedding.add(Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size))
        self.to_patch_embedding.add(layer_norm(patch_dim))
        self.to_patch_embedding.add(dense(dim, patch_dim))
        self.to_patch_embedding.add(layer_norm(dim))

        self.pos_embedding = initializer_((1, num_patches + 1, dim), 'normal')
        self.cls_token = initializer_((1, 1, dim), 'normal')
        self.dropout = dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate)

        self.pool = pool
        self.to_latent = identity()

        self.mlp_head = Layers()
        self.mlp_head.add(layer_norm(dim))
        self.mlp_head.add(dense(num_classes, dim))
        
        self.param = Module.param
        self.training = True
    
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
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, self.training)

        x = self.transformer(x, self.training)

        x = tf.reduce_mean(x, axis=-1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)