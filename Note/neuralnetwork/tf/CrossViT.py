import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.dropout import dropout
from Note.nn.layer.identity import identity
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.initializer import initializer_
from Note.nn.Layers import Layers
from Note.nn.Model import Model

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# feedforward

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

# attention

class Attention:
    def __init__(self, dim, heads = 8, dim_head = 64, dropout_rate = 0.):
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = layer_norm(dim)
        self.attend = tf.nn.softmax
        self.dropout = dropout(dropout_rate)

        self.to_q = dense(inner_dim, dim, use_bias = False)
        self.to_kv = dense(inner_dim * 2, dim, use_bias = False)

        self.to_out = Layers()
        self.to_out.add(dense(dim, inner_dim))
        self.to_out.add(dropout(dropout_rate))

    def __call__(self, x, context = None, kv_include_self = False, training=True):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = tf.concat((x, context), axis = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *tf.split(self.to_kv(context), 2, axis=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, training)

# transformer encoder, for small and large patches

class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        self.layers = []
        self.norm = layer_norm(dim)
        for _ in range(depth):
            self.layers.append([
                Attention(dim, heads = heads, dim_head = dim_head, dropout_rate = dropout),
                FeedForward(dim, mlp_dim, dropout_rate = dropout)
            ])

    def __call__(self, x, training=True):
        for attn, ff in self.layers:
            x = attn(x, training=training) + x
            x = ff(x, training) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut:
    def __init__(self, dim_in, dim_out, fn):
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = dense(dim_out, dim_in) if need_projection else identity()
        self.project_out = dense(dim_in, dim_out) if need_projection else identity()

    def __call__(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer:
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                ProjectInOut(sm_dim, lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout_rate = dropout)),
                ProjectInOut(lg_dim, sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout_rate = dropout))
            ])

    def __call__(self, sm_tokens, lg_tokens, training=True):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True, training=training) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True, training=training) + lg_cls

        sm_tokens = tf.concat((sm_cls, sm_patch_tokens), axis = 1)
        lg_tokens = tf.concat((lg_cls, lg_patch_tokens), axis = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder:
    def __init__(
        self,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ])

    def __call__(self, sm_tokens, lg_tokens, training=True):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens, training), lg_enc(lg_tokens, training)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens, training)

        return sm_tokens, lg_tokens

# patch-based image to token embedder

class ImageEmbedder:
    def __init__(
        self,
        dim,
        image_size,
        patch_size,
        dropout_rate = 0.,
        channels = 3
    ):
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = Layers()
        self.to_patch_embedding.add(Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size))
        self.to_patch_embedding.add(layer_norm(patch_dim))
        self.to_patch_embedding.add(dense(dim, patch_dim))
        self.to_patch_embedding.add(layer_norm(dim))

        self.pos_embedding = initializer_((1, num_patches + 1, dim), 'normal')
        self.cls_token = initializer_((1, 1, dim), 'normal')
        self.dropout = dropout(dropout_rate)

    def __call__(self, img, training=True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = tf.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x, training)

# cross ViT class

class CrossViT(Model):
    def __init__(
        self,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        
        self.sm_dim = sm_dim
        self.lg_dim = lg_dim
        
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, channels= channels, image_size = image_size, patch_size = sm_patch_size, dropout_rate = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, channels = channels, image_size = image_size, patch_size = lg_patch_size, dropout_rate = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = Layers()
        self.sm_mlp_head.add(layer_norm(sm_dim))
        self.sm_mlp_head.add(dense(num_classes, sm_dim))
        self.lg_mlp_head = Layers()
        self.lg_mlp_head.add(layer_norm(lg_dim))
        self.lg_mlp_head.add(dense(num_classes, lg_dim))
        
        self.training = True
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.sm_mlp_head_=self.sm_mlp_head.layer[-1]
            self.sm_mlp_head.layer[-1]=dense(classes, self.sm_dim)
            self.lg_mlp_head_=self.lg_mlp_head.layer[-1]
            self.lg_mlp_head.layer[-1]=dense(classes, self.lg_dim)
            param.extend(self.sm_mlp_head.layer[-1].param)
            param.extend(self.lg_mlp_head.layer[-1].param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.lg_mlp_head.layer[-1].param):]
            del self.param_[-len(self.sm_mlp_head.layer[-1].param):]
            self.param_.extend(self.sm_mlp_head.layer[-1].param)
            self.param_.extend(self.lg_mlp_head.layer[-1].param)
            self.param=self.param_
        else:
            self.sm_mlp_head.layer[-1],self.sm_mlp_head_=self.sm_mlp_head_,self.sm_mlp_head.layer[-1]
            self.lg_mlp_head.layer[-1],self.lg_mlp_head_=self.lg_mlp_head_,self.lg_mlp_head.layer[-1]
            del self.param_[-len(self.lg_mlp_head.layer[-1].param):]
            del self.param_[-len(self.sm_mlp_head.layer[-1].param):]
            self.param_.extend(self.sm_mlp_head.layer[-1].param)
            self.param_.extend(self.lg_mlp_head.layer[-1].param)
            self.param=self.param_
        return

    def __call__(self, data):
        sm_tokens = self.sm_image_embedder(data, self.training)
        lg_tokens = self.lg_image_embedder(data, self.training)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens, self.training)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits