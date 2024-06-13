import tensorflow as tf
from Note import nn

from einops import rearrange, repeat, reduce
from einops.layers.tensorflow import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.layer_norm(dim)
        self.attend = tf.nn.softmax
        self.dropout = nn.dropout(dropout_rate)

        self.to_qkv = nn.dense(inner_dim * 3, dim, use_bias = False)

        self.to_out = nn.Sequential()
        if project_out:
            self.to_out.add(nn.dense(dim, inner_dim))
            self.to_out.add(nn.dropout(dropout_rate))
        else:
            self.to_out.add(nn.identity())

    def __call__(self, x, training):
        x = self.norm(x)
        qkv = tf.split(self.to_qkv(x), 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn, training)

        out = tf.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out, training)

class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout_rate = 0.):
        self.norm = nn.layer_norm(dim)
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
        return self.norm(x)

class ViViT(nn.Model):
    def __init__(
        self,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout_rate = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size
        self.dim = dim

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential()
        self.to_patch_embedding.add(Rearrange('b (f pf) (h p1) (w p2) c -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size))
        self.to_patch_embedding.add(nn.layer_norm(patch_dim))
        self.to_patch_embedding.add(nn.dense(dim, patch_dim))
        self.to_patch_embedding.add(nn.layer_norm(dim))

        self.pos_embedding = nn.initializer_((1, num_frame_patches, num_image_patches, dim), 'normal')
        self.dropout = nn.dropout(emb_dropout)

        self.spatial_cls_token = nn.initializer_((1, 1, dim), 'normal') if not self.global_average_pool else None
        self.temporal_cls_token = nn.initializer_((1, 1, dim), 'normal') if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout_rate)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout_rate)

        self.pool = pool
        self.to_latent = nn.identity()

        self.head = self.dense(num_classes, dim)
        
        self.training = True

    def __call__(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = tf.concat((spatial_cls_tokens, x), axis = 2)

        x = self.dropout(x, self.training)

        x = rearrange(x, 'b f n d -> (b f) n d')

        # attend across space

        x = self.spatial_transformer(x, self.training)

        x = rearrange(x, '(b f) n d -> b f n d', b = b)

        # excise out the spatial cls tokens or average pool for temporal attention

        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = tf.concat((temporal_cls_tokens, x), axis = 1)

        # attend across time

        x = self.temporal_transformer(x, self.training)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        x = self.to_latent(x)
        return self.head(x)