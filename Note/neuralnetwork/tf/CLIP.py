import tensorflow as tf
from Note import nn
import numpy as np
from typing import Tuple, Union


class Bottleneck:
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.conv2d(planes, 1, inplanes, use_bias=False)
        self.bn1 = nn.batch_norm(planes)
        self.relu1 = tf.nn.relu

        self.conv2 = nn.conv2d(planes, 3, planes, padding=1, use_bias=False)
        self.bn2 = nn.batch_norm(planes)
        self.relu2 = tf.nn.relu

        self.avgpool = nn.avg_pool2d(stride, stride, 'VALID') if stride > 1 else nn.identity()

        self.conv3 = nn.conv2d(planes * self.expansion, 1, planes, use_bias=False)
        self.bn3 = nn.batch_norm(planes * self.expansion)
        self.relu3 = tf.nn.relu

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential()
            self.downsample.add(nn.avg_pool2d(stride, stride, 'VALID'))
            self.downsample.add(nn.conv2d(planes * self.expansion, 1, inplanes, strides=1, use_bias=False))
            self.downsample.add(nn.batch_norm(planes * self.expansion))

    def __call__(self, x):
        nn.identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            nn.identity = self.downsample(x)

        out += nn.identity
        out = self.relu3(out)
        return out


class AttentionPool2d:
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        self.positional_embedding = tf.Variable(tf.random.normal([spacial_dim ** 2 + 1, embed_dim]) / embed_dim ** 0.5)
        self.k_proj = nn.dense(embed_dim, embed_dim)
        self.q_proj = nn.dense(embed_dim, embed_dim)
        self.v_proj = nn.dense(embed_dim, embed_dim)
        self.c_proj = nn.dense(output_dim or embed_dim, embed_dim)
        self.num_heads = num_heads
        nn.Model.param.append(self.positional_embedding)

    def __call__(self, x):
        shape = x.shape
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        new_shape = (batch_size, height * width, channels)
        x = tf.transpose(tf.reshape(x, new_shape), (1, 0, 2))
        x = tf.concat([tf.reduce_mean(x, axis=0, keepdims=True), x], axis=0)  # (HW+1)NC
        x = x + tf.cast(self.positional_embedding[:, None, :], x.dtype)  # (HW+1)NC
        tgt_len, bsz, embed_dim = x.shape
        query=self.q_proj(x[:1])
        key=self.k_proj(x)
        value=self.v_proj(x)
        query = tf.reshape(query, [bsz, 1, self.num_heads, -1])
        query = tf.transpose(query, [0, 2, 1, 3])
        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(embed_dim)))
        key = tf.reshape(key, [bsz, tgt_len, self.num_heads, -1])
        key = tf.transpose(key, [0, 2, 3, 1])
        value = tf.reshape(value, [bsz, tgt_len, self.num_heads, -1])
        value = tf.transpose(value, [0, 2, 1, 3])
        qk = tf.matmul(query, key)
        w = tf.nn.softmax(qk)
        wv = tf.reshape(tf.transpose(tf.matmul(w, value), [0, 2, 1, 3]), [1, bsz, -1])
        x = self.c_proj(wv)
        return tf.squeeze(x, 0)


class ModifiedResNet:
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.conv2d(width // 2, input_size=3, kernel_size=3, strides=2, padding=1, use_bias=False)
        self.bn1 = nn.batch_norm(width // 2)
        self.relu1 = tf.nn.relu
        self.conv2 = nn.conv2d(width // 2, input_size=width // 2, kernel_size=3, padding=1, use_bias=False)
        self.bn2 = nn.batch_norm(width // 2)
        self.relu2 = tf.nn.relu
        self.conv3 = nn.conv2d(width, input_size=width // 2, kernel_size=3, padding=1, use_bias=False)
        self.bn3 = nn.batch_norm(width)
        self.relu3 = tf.nn.relu
        self.avgpool = nn.avg_pool2d(2, 2, 'VALID')

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = nn.Sequential()
        layers.add(Bottleneck(self._inplanes, planes, stride))

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.add(Bottleneck(self._inplanes, planes))

        return layers

    def __call__(self, x):
        def stem(x):
            x = self.conv1(x)
            x = self.relu1(self.bn1(x))
            x = self.conv2(x)
            x = self.relu2(self.bn2(x))
            x = self.conv3(x)
            x = self.relu3(self.bn3(x))
            x = self.avgpool(x)
            return x

        x = tf.cast(x, self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm:
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, input_size):
        self.layer_norm = nn.layer_norm(input_size)

    def __call__(self, x):
        orig_type = x.dtype
        ret = self.layer_norm(tf.cast(x, tf.float32))
        return tf.cast(ret, orig_type)


class QuickGELU:
    def __call__(self, x):
        return x * tf.nn.sigmoid(1.702 * x)


class ResidualAttentionBlock:
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        self.attn = nn.multihead_attention(n_head, d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential()
        self.mlp.add(nn.dense(d_model * 4, d_model))
        self.mlp.add(QuickGELU())
        self.mlp.add(nn.dense(d_model, d_model * 4))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = tf.cast(self.attn_mask, x.dtype) if self.attn_mask is not None else None
        return self.attn(x, mask=self.attn_mask)[0]
    
    def init(self,attn_std, proj_std, fc_std):
        shape = self.attn.query.weight.shape
        self.attn.query.weight.assign(tf.Variable(tf.random.normal(shape,stddev=attn_std)))
        shape = self.attn.key.weight.shape
        self.attn.key.weight.assign(tf.Variable(tf.random.normal(shape,stddev=attn_std)))
        shape = self.attn.value.weight.shape
        self.attn.value.weight.assign(tf.Variable(tf.random.normal(shape,stddev=attn_std)))
        shape = self.attn.out.weight.shape
        self.attn.out.weight.assign(tf.Variable(tf.random.normal(shape,stddev=proj_std)))
        shape = self.mlp.layer[0].weight.shape
        self.mlp.layer[0].weight.assign(tf.Variable(tf.random.normal(shape,stddev=fc_std)))
        shape = self.mlp.layer[2].weight.shape
        self.mlp.layer[2].weight.assign(tf.Variable(tf.random.normal(shape,stddev=proj_std)))

    def __call__(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer:
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential()
        for _ in range(layers):
            self.resblocks.add(ResidualAttentionBlock(width, heads, attn_mask))

    def __call__(self, x):
        return self.resblocks(x)


class VisionTransformer:
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.conv2d(width, input_size=3, kernel_size=patch_size, strides=patch_size, use_bias=False)

        scale = width ** -0.5
        self.class_embedding = tf.Variable(scale * tf.random.normal([width]))
        self.positional_embedding = tf.Variable(scale * tf.random.normal((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = tf.Variable(scale * tf.random.normal(width, output_dim))
        nn.Model.param.append(self.class_embedding)
        nn.Model.param.append(self.positional_embedding)
        nn.Model.param.append(self.proj)

    def __call__(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = tf.reshape(x, [x.shape[0], x.shape[1], -1])  # shape = [*, width, grid ** 2]
        x = tf.transpose(x, (0, 2, 1))  # shape = [*, grid ** 2, width]
        x = tf.concat([tf.cast(self.class_embedding, x.dtype) + tf.zeros([x.shape[0], 1, x.shape[-1]], dtype=x.dtype), x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + tf.cast(self.positional_embedding, x.dtype)
        x = self.ln_pre(x)

        x = tf.transpose(x, (1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = tf.transpose(x, (1, 0, 2))  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = tf.matmul(x, self.proj)

        return x


class CLIP(nn.Model):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.initializer((vocab_size, transformer_width), 
                                            ['normal', 0.0, 0.02],
                                                'float32')
        self.positional_embedding = nn.initializer((self.context_length, transformer_width), 
                                                 ['normal', 0.0, 0.01],
                                                 'float32')
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.initializer((transformer_width, embed_dim), 
                                            ['normal', 0.0,self.transformer.width ** -0.5], 
                                            'float32')
        self.logit_scale = tf.Variable(tf.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.input_size ** -0.5
                shape = self.visual.attnpool.q_proj.weight.shape
                self.visual.attnpool.q_proj.weight.assign(tf.Variable(tf.random.normal(shape,stddev=std)))
                shape = self.visual.attnpool.k_proj.weight.shape
                self.visual.attnpool.k_proj.weight.assign(tf.Variable(tf.random.normal(shape,stddev=std)))
                shape = self.visual.attnpool.v_proj.weight.shape
                self.visual.attnpool.v_proj.weight.assign(tf.Variable(tf.random.normal(shape,stddev=std)))
                shape = self.visual.attnpool.c_proj.weight.shape
                self.visual.attnpool.c_proj.weight.assign(tf.Variable(tf.random.normal(shape,stddev=std)))

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks.layer:
            block.init(attn_std, proj_std, fc_std)

    def build_attention_mask(self):
        mask = tf.ones((self.context_length, self.context_length))
        mask = tf.linalg.band_part(mask, 0, -1) # zero out the upper diagonal
        mask = mask * -1e9 # fill with -1e9
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(tf.cast(image, self.dtype))

    def encode_text(self, text):
        x = tf.cast(tf.gather(self.token_embedding, text), self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + tf.cast(self.positional_embedding, self.dtype)
        x = tf.transpose(x, (1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = tf.transpose(x, (1, 0, 2))  # LND -> NLD
        x = tf.cast(self.ln_final(x), self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = tf.matmul(tf.gather_nd(x, tf.stack([tf.range(x.shape[0], dtype='int32'), 
                        tf.argmax(text, axis=-1, output_type='int32')], axis=1)), self.text_projection)

        return x

    def __call__(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / tf.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / tf.norm(text_features, axis=1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_image = tf.matmul(logit_scale * image_features, tf.transpose(text_features))
        logits_per_text = tf.transpose(logits_per_image)

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def convert_weights(self):
        """Convert applicable model parameters to fp16"""
        self.cast_param('dense_weight', 'float16')
        self.cast_param('dense_bias', 'float16')
        self.cast_param('conv2d_weight', 'float16')
        self.cast_param('conv2d_bias', 'float16')