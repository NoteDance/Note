import tensorflow as tf
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.dense import dense
from Note.nn.layer.dropout import dropout
from Note.nn.layer.identity import identity
from Note.nn.initializer import initializer_
from Note.nn.Layers import Layers
from Note.nn.Module import Module


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward:
    def __init__(self, dim, hidden_dim, drop_rate = 0.):
        self.net = Layers()
        self.net.add(layer_norm(dim))
        self.net.add(dense(hidden_dim, dim))
        self.net.add(tf.nn.gelu)
        self.net.add(dropout(drop_rate))
        self.net.add(dense(dim, hidden_dim))
        self.net.add(dropout(drop_rate))

    def __call__(self, x, train_flag=True):
        return self.net(x, train_flag)


class Attention:
    def __init__(self, dim, heads = 8, dim_head = 64, drop_rate = 0.):
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = layer_norm(dim)

        self.attend = tf.nn.softmax
        self.dropout = dropout(drop_rate)

        self.to_qkv = dense(inner_dim * 3, dim, use_bias = False)
        
        if project_out:
            self.to_out = Layers()
            self.to_out.add(dense(dim, inner_dim))
            self.to_out.add(dropout(drop_rate))
        else:
            self.to_out = identity()

    def __call__(self, x, train_flag=True):
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
        attn = self.dropout(attn, train_flag)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 1, 3, 2])
        out = tf.reshape(out, shape=[-1, n, h*d])
        return self.to_out(out)


class Transformer:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        self.norm = layer_norm(dim)
        self.layers = []
        for _ in range(depth):
            self.layers.append([Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                                FeedForward(dim, mlp_dim, dropout = dropout)])

    def __call__(self, x, train_flag=True):
        for attn, ff in self.layers:
            x = attn(x, train_flag) + x
            x = ff(x, train_flag) + x

        return self.norm(x)


class ViT:
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        Module.init()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.p1, self.p2 = patch_height, patch_width
        self.dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = Layers()
        self.to_patch_embedding.add(layer_norm(patch_dim))
        self.to_patch_embedding.add(dense(dim, patch_dim))
        self.to_patch_embedding.add(layer_norm(dim))

        self.pos_embedding = initializer_((1, num_patches + 1, dim), 'normal', 'float32')
        self.cls_token = initializer_((1, 1, dim), 'normal', 'float32')
        self.dropout = dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = identity()

        self.mlp_head = dense(num_classes, dim)
        
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.opt=tf.keras.optimizers.Adam()
        self.param=Module.param
        self.km=0
        
        
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.mlp_head_=self.mlp_head
            self.mlp_head=dense(classes, self.dim)
            param.extend(self.mlp_head.param)
            self.param=param
            self.opt.lr=lr
        elif flag==1:
            del self.param_[-len(self.mlp_head.param):]
            self.param_.extend(self.mlp_head.param)
            self.param=self.param_
        else:
            self.mlp_head,self.mlp_head_=self.mlp_head_,self.mlp_head
            del self.param_[-len(self.mlp_head.param):]
            self.param_.extend(self.mlp_head.param)
            self.param=self.param_
        return


    def fp(self, data):
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
        x = self.dropout(x, self.km)

        x = self.transformer(x, self.km)

        x = tf.reduce_mean(x, axis = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return tf.nn.softmax(self.mlp_head(x))

    
    def loss(self,output,labels):
        loss=self.loss_object(labels,output)
        return loss
