import tensorflow as tf
from Note import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward:
    def __init__(self, dim, hidden_dim, drop_rate = 0.):
        self.net = nn.Sequential()
        self.net.add(nn.layer_norm(dim))
        self.net.add(nn.dense(hidden_dim, dim))
        self.net.add(tf.nn.gelu)
        self.net.add(nn.dropout(drop_rate))
        self.net.add(nn.dense(dim, hidden_dim))
        self.net.add(nn.dropout(drop_rate))

    def __call__(self, x):
        return self.net(x)


class Attention:
    def __init__(self, dim, heads = 8, dim_head = 64, drop_rate = 0.):
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.layer_norm(dim)

        self.attend = tf.nn.softmax
        self.dropout = nn.dropout(drop_rate)

        self.to_qkv = nn.dense(inner_dim * 3, dim, use_bias = False)
        
        if project_out:
            self.to_out = nn.Sequential()
            self.to_out.add(nn.dense(dim, inner_dim))
            self.to_out.add(nn.dropout(drop_rate))
        else:
            self.to_out = nn.identity()

    def __call__(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        b = q.shape[0]
        h = self.heads
        n = q.shape[1]
        d = q.shape[2] // self.heads
        q = tf.reshape(q, (b, h, n, d))
        k = tf.reshape(k, (b, h, n, d))
        v = tf.reshape(v, (b, h, n, d))

        dots = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 1, 3, 2])
        out = tf.reshape(out, shape=[-1, n, h*d])
        return self.to_out(out)


class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        self.norm = nn.layer_norm(dim)
        self.layers = []
        for _ in range(depth):
            self.layers.append([Attention(dim, heads = heads, dim_head = dim_head, drop_rate = dropout),
                                FeedForward(dim, mlp_dim, drop_rate = dropout)])

    def __call__(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, drop_rate = 0., emb_dropout = 0.):
        super().__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.p1, self.p2 = patch_height, patch_width
        self.dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential()
        self.to_patch_embedding.add(nn.layer_norm(patch_dim))
        self.to_patch_embedding.add(nn.dense(dim, patch_dim))
        self.to_patch_embedding.add(nn.layer_norm(dim))

        self.pos_embedding = nn.initializer((1, num_patches + 1, dim), 'normal', 'float32')
        self.cls_token = nn.initializer((1, 1, dim), 'normal', 'float32')
        self.dropout = nn.dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, drop_rate)

        self.pool = pool
        self.to_latent = nn.identity()

        self.head = self.dense(num_classes, dim)

    def __call__(self, data):
        b = data.shape[0]
        h = data.shape[1] // self.p1
        w = data.shape[2] // self.p2
        c = data.shape[3]
        data = tf.reshape(data, (b, h * w, self.p1 * self.p2 * c))
        x = self.to_patch_embedding(data)
        b, n, _ = x.shape

        cls_tokens = tf.tile(self.cls_token, multiples=[b, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = tf.reduce_mean(x, axis = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return tf.nn.softmax(self.mlp_head(x))