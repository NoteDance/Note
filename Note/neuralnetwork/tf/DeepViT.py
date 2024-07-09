import tensorflow as tf
from Note import nn

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

class FeedForward:
    def __init__(self, dim, hidden_dim, dropout_rate = 0.):
        self.net = nn.Sequential()
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
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.layer_norm(dim)
        self.to_qkv = nn.dense(inner_dim * 3, dim, use_bias = False)

        self.dropout = nn.dropout(dropout_rate)

        self.reattn_weights = nn.initializer((heads, heads), 'normal')

        self.reattn_norm = nn.Sequential()
        self.reattn_norm.add(Rearrange('b h i j -> b i j h'))
        self.reattn_norm.add(nn.layer_norm(heads))
        self.reattn_norm.add(Rearrange('b i j h -> b h i j'))

        self.to_out = nn.Sequential()
        self.to_out.add(nn.dense(dim, inner_dim))
        self.to_out.add(nn.dropout(dropout_rate))

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

class DeepViT(nn.Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout_rate = 0., emb_dropout = 0.):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.dim = dim

        self.to_patch_embedding = nn.Sequential()
        self.to_patch_embedding.add(Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size))
        self.to_patch_embedding.add(nn.layer_norm(patch_dim))
        self.to_patch_embedding.add(nn.dense(dim, patch_dim))
        self.to_patch_embedding.add(nn.layer_norm(dim))

        self.pos_embedding = nn.initializer((1, num_patches + 1, dim), 'normal')
        self.cls_token = nn.initializer((1, 1, dim), 'normal')
        self.dropout = nn.dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate)

        self.pool = pool
        self.to_latent = nn.identity()

        self.mlp_head = nn.Sequential()
        self.mlp_head.add(nn.layer_norm(dim))
        self.mlp_head.add(nn.dense(num_classes, dim))
        
        self.training = True
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
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