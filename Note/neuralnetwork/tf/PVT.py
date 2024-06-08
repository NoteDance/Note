import tensorflow as tf
from Note import nn
from functools import partial
import math


class Mlp:
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0., linear=False):
        nn.Model.add()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.dense(hidden_features, in_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = nn.dense(out_features, hidden_features)
        self.drop = nn.dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = tf.nn.relu
        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        elif isinstance(l, nn.conv2d):
            fan_out = l.kernel_size[0] * l.kernel_size[1] * l.output_size
            fan_out //= l.groups
            if l.groups==1:
                l.weight.assign(nn.initializer(l.weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))
            else:
                for weight in l.weight:
                    weight.assign(nn.initializer(weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))

    def __call__(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        nn.Model.add()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.dense(dim, dim, use_bias=qkv_bias)
        self.kv = nn.dense(dim * 2, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.conv2d(dim, sr_ratio, input_size=dim, strides=sr_ratio)
                self.norm = nn.layer_norm(dim)
        else:
            self.pool = nn.adaptive_avg_pooling2d(7)
            self.sr = nn.conv2d(dim, 1, input_size=dim, strides=1)
            self.norm = nn.layer_norm(dim)
            self.act = tf.nn.gelu
        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        elif isinstance(l, nn.conv2d):
            fan_out = l.kernel_size[0] * l.kernel_size[1] * l.output_size
            fan_out //= l.groups
            if l.groups==1:
                l.weight.assign(nn.initializer(l.weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))
            else:
                for weight in l.weight:
                    weight.assign(nn.initializer(weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))

    def __call__(self, x, H, W):
        B, N, C = x.shape
        q = tf.transpose(tf.reshape(self.q(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = tf.reshape(tf.transpose(x, (0, 2, 1)), (B, H, W, C))
                x_ = tf.reshape(self.sr(x_), (B, -1, C))
                x_ = self.norm(x_)
                kv = tf.transpose(tf.reshape(self.kv(x_), (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
            else:
                kv = tf.transpose(tf.reshape(self.kv(x), (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        else:
            x_ = tf.reshape(x, (B, H, W, C))
            x_ = tf.reshape(self.sr(self.pool(x_)), (B, -1, C))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = tf.transpose(tf.reshape(self.kv(x_), (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block:

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, sr_ratio=1, linear=False):
        nn.Model.add()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        elif isinstance(l, nn.conv2d):
            fan_out = l.kernel_size[0] * l.kernel_size[1] * l.output_size
            fan_out //= l.groups
            if l.groups==1:
                l.weight.assign(nn.initializer(l.weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))
            else:
                for weight in l.weight:
                    weight.assign(nn.initializer(weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))

    def __call__(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed:
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, name=None):
        nn.Model.add()
        nn.Model.name_ = name
        
        img_size = nn.to_2tuple(img_size)
        patch_size = nn.to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.conv2d(embed_dim, patch_size, input_size=in_chans, strides=stride, 
                              padding=(patch_size[0] // 2, patch_size[1] // 2)
                              )
        self.norm = nn.layer_norm(embed_dim)

        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        elif isinstance(l, nn.conv2d):
            fan_out = l.kernel_size[0] * l.kernel_size[1] * l.output_size
            fan_out //= l.groups
            if l.groups==1:
                l.weight.assign(nn.initializer(l.weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))
            else:
                for weight in l.weight:
                    weight.assign(nn.initializer(weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, _ = x.shape
        x = tf.reshape(x, (B, H*W, -1))
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.layer_norm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        nn.Model.add()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = tf.linspace(0., drop_path_rate, sum(depths))  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i==0:
                freeze_name=f"patch_embed{i + 1}"
            else:
                freeze_name=None
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], name=freeze_name)

            block = [Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])]
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = self.dense(num_classes, embed_dims[3]) if num_classes > 0 else nn.identity()

        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        elif isinstance(l, nn.conv2d):
            fan_out = l.kernel_size[0] * l.kernel_size[1] * l.output_size
            fan_out //= l.groups
            if l.groups==1:
                l.weight.assign(nn.initializer(l.weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))
            else:
                for weight in l.weight:
                    weight.assign(nn.initializer(weight.shape, ['normal', 0, math.sqrt(2.0 / fan_out)]))

    def freeze_patch_emb(self):
        self.freeze('patch_embed1')

    def no_weight_decay(self):
        return ['pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token']  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.dense(num_classes, self.embed_dim) if num_classes > 0 else nn.identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = tf.reshape(x, (B, H, W, -1))
            
        return tf.reduce_mean(x, axis=1)

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv:
    def __init__(self, dim=768):
        self.dwconv = nn.conv2d(dim, 3, dim, 1, groups=dim, padding=1, use_bias=True)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        x = tf.reshape(x, (B, H, W, C))
        x = self.dwconv(x)
        B, H, W, _ = x.shape
        x = tf.reshape(x, (B, H*W, -1))

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def pvt_v2_b0(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b1(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b2(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model


def pvt_v2_b3(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b4(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b5(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b2_li(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    return model