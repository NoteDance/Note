""" EfficientViT (by MSRA)

Paper: `EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention`
    - https://arxiv.org/abs/2305.07027

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/EfficientViT
"""

__all__ = ['EfficientVitMsra']
import itertools
from typing import Optional

import tensorflow as tf
from Note import nn


class ConvNorm:
    def __init__(self, in_chs, out_chs, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        self.conv = nn.conv2d(out_chs, ks, in_chs, stride, pad, dilations=dilation, groups=groups, use_bias=False)
        self.bn = nn.batch_norm(out_chs)

    def __call__(self):
        c, bn = self.conv, self.bn
        c.weight._trainable=False
        bn.gamma._trainable=False
        bn.beta._trainable=False
        w = bn.gamma / (bn.moving_variance + bn.epsilon)**0.5
        w = c.weight * w[None, None, None, :]
        b = bn.beta - bn.moving_mean * bn.gamma / \
                (bn.moving_variance + bn.epsilon)**0.5
        m = nn.conv2d(
            w.size(0), w.shape[2:], w.size(1) * self.conv.groups,
            strides=self.conv.stride, padding=self.conv.padding, dilations=self.conv.dilation, groups=self.conv.groups)
        c.weight._trainable=True
        bn.gamma._trainable=True
        bn.beta._trainable=True
        m.weight.assign(w)
        m.bias.assign(b)
        return m


class NormLinear:
    def __init__(self, in_features, out_features, bias=True, std=0.02, drop=0.):
        self.bn = nn.batch_norm(in_features)
        self.drop = nn.dropout(drop)
        self.linear = nn.dense(out_features, in_features, use_bias=bias)

        nn.trunc_normal_(self.linear.weight, std=std)

    def __call__(self):
        bn, linear = self.bn, self.linear
        linear.weight._trainable=False
        self.linear.weight._trainable=False
        self.linear.bias._trainable=False
        bn.gamma._trainable=False
        bn.beta._trainable=False
        w = bn.gamma / (bn.moving_variance + bn.epsilon)**0.5
        b = bn.beta - self.bn.moving_mean * \
                self.bn.gamma / (bn.moving_variance + bn.epsilon)**0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = tf.matmul(b, self.linear.weight, transpose_b=True)
        else:
            b = tf.reshape(tf.matmul(linear.weight, b[:, None]), (-1)) + self.linear.bias
        linear.weight._trainable=True
        self.linear.weight._trainable=True
        self.linear.bias._trainable=True
        bn.gamma._trainable=True
        bn.beta._trainable=True
        m = nn.dense(w.shape[1], w.shape[0])
        m.weight.assign(w)
        m.bias.assign(b)
        return m


class PatchMerging:
    def __init__(self, dim, out_dim):
        hid_dim = int(dim * 4)
        self.conv1 = ConvNorm(dim, hid_dim, 1, 1, 0)
        self.act = tf.nn.relu
        self.conv2 = ConvNorm(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = nn.SqueezeExcite(hid_dim, .25)
        self.conv3 = ConvNorm(hid_dim, out_dim, 1, 1, 0)

    def __call__(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class ResidualDrop:
    def __init__(self, m, drop=0.):
        self.m = m
        self.drop = drop
        self.training = True
        nn.Model.layer_list.append(self)

    def __call__(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * tf.stop_gradient(tf.cast(tf.greater_equal(tf.random.uniform((
                x.shape[0], 1, 1, 1)), self.drop), x.dtype) / (1 - self.drop))
        else:
            return x + self.m(x)


class ConvMlp:
    def __init__(self, ed, h):
        self.pw1 = ConvNorm(ed, h)
        self.act = tf.nn.relu
        self.pw2 = ConvNorm(h, ed, bn_weight_init=0)

    def __call__(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention:
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            kernels=(5, 5, 5, 5),
    ):
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(ConvNorm(dim // (num_heads), self.key_dim * 2 + self.val_dim))
            dws.append(ConvNorm(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim))
        self.qkvs = qkvs
        self.dws = dws
        self.proj = nn.Sequential()
        self.proj.add(tf.nn.relu)
        self.proj.add(ConvNorm(self.val_dim * num_heads, dim, bn_weight_init=0))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(tf.zeros((num_heads, len(attention_offsets))), name='attention_biases')
        self.attention_bias_idxs = tf.reshape(tf.constant(idxs, dtype=tf.int64), (N, N))
        self.training =True
        nn.Model.layer_list.append(self)

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.training:
            trainingab = self.attention_biases[:, self.attention_bias_idxs]
            if hasattr(self, 'ab'):
                del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = tf.split(x, len(self.qkvs), axis=-1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = tf.split(tf.reshape(feat, (B, -1, H, W)), [self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = tf.reshape(q, q.shape[0], q.shape[1], q.shape[2]*q.shape[3]), 
            tf.reshape(k, k.shape[0], k.shape[1], k.shape[2]*k.shape[3]), 
            tf.reshape(v, v.shape[0], v.shape[1], v.shape[2]*v.shape[3]) # B, C/h, N
            attn = (
                tf.matmul(tf.transpose(q, (0, 1, 3, 2)), k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = tf.nn.softmax(attn, axis=-1) # BNN
            feat = tf.reshape(tf.matmul(v, tf.transpose(attn, (0, 2, 1))), (B, H, W, self.d)) # BHWC
            feats_out.append(feat)
        x = self.proj(tf.concat(feats_out, -1))
        return x


class LocalWindowAttention:
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=(5, 5, 5, 5),
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            dim, key_dim, num_heads,
            attn_ratio=attn_ratio,
            resolution=window_resolution,
            kernels=kernels,
        )

    def __call__(self, x):
        H = W = self.resolution
        B, H_, W_, C = x.shape
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W % self.window_resolution) % self.window_resolution
            paddings = tf.constant([[0, 0], [0, pad_b], [0, pad_r], [0, 0]])
            x = tf.pad(x, paddings)

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = tf.transpose(tf.reshape(x, (B, nH, self.window_resolution, nW, self.window_resolution, C)), (0, 1, 3, 2, 4, 5))
            x = tf.reshape(x, (B * nH * nW, self.window_resolution, self.window_resolution, C))
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = tf.reshape(tf.transpose(x, (0, 2, 3, 1)), (B, nH, nW, self.window_resolution, self.window_resolution, C))
            x = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (B, pH, pW, C))
            x = x[:, :H, :W]
        return x


class EfficientVitBlock:
    """ A basic EfficientVit building block.

    Args:
        dim (int): Number of input channels.
        key_dim (int): Dimension for query and key in the token mixer.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=[5, 5, 5, 5],
    ):

        self.dw0 = ResidualDrop(ConvNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.))
        self.ffn0 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

        self.mixer = ResidualDrop(
            LocalWindowAttention(
                dim, key_dim, num_heads,
                attn_ratio=attn_ratio,
                resolution=resolution,
                window_resolution=window_resolution,
                kernels=kernels,
            )
        )

        self.dw1 = ResidualDrop(ConvNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.))
        self.ffn1 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

    def __call__(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientVitStage:
    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            downsample=('', 1),
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=[5, 5, 5, 5],
            depth=1,
    ):
        if downsample[0] == 'subsample':
            self.resolution = (resolution - 1) // downsample[1] + 1
            self.downsample = nn.Sequential()
            self.downsample.add(ResidualDrop(ConvNorm(in_dim, in_dim, 3, 1, 1, groups=in_dim)))
            self.downsample.add(ResidualDrop(ConvMlp(in_dim, int(in_dim * 2))))
            self.downsample.add(PatchMerging(in_dim, out_dim))
            self.downsample.add(ResidualDrop(ConvNorm(out_dim, out_dim, 3, 1, 1, groups=out_dim)))
            self.downsample.add(ResidualDrop(ConvMlp(out_dim, int(out_dim * 2))))
        else:
            assert in_dim == out_dim
            self.downsample = nn.identity()
            self.resolution = resolution

        blocks = []
        for d in range(depth):
            blocks.append(EfficientVitBlock(out_dim, key_dim, num_heads, attn_ratio, self.resolution, window_resolution, kernels))
        self.blocks = nn.Sequential()
        self.blocks.add(blocks)

    def __call__(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class PatchEmbedding(nn.Sequential):
    def __init__(self, in_chans, dim):
        super().__init__()
        self.add(ConvNorm(in_chans, dim // 8, 3, 2, 1))
        self.add(tf.nn.relu)
        self.add(ConvNorm(dim // 8, dim // 4, 3, 2, 1))
        self.add(tf.nn.relu)
        self.add(ConvNorm(dim // 4, dim // 2, 3, 2, 1))
        self.add(tf.nn.relu)
        self.add(ConvNorm(dim // 2, dim, 3, 2, 1))
        self.patch_size = 16


class EfficientVitMsra(nn.Model):
    def __init__(
            self,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            embed_dim=(64, 128, 192),
            key_dim=(16, 16, 16),
            depth=(1, 2, 3),
            num_heads=(4, 4, 4),
            window_size=(7, 7, 7),
            kernels=(5, 5, 5, 5),
            down_ops=(('', 1), ('subsample', 2), ('subsample', 2)),
            global_pool='avg',
            drop_rate=0.,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_chans, embed_dim[0])
        stride = self.patch_embed.patch_size
        resolution = img_size // self.patch_embed.patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]

        # Build EfficientVit blocks
        self.feature_info = []
        stages = []
        pre_ed = embed_dim[0]
        for i, (ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            stage = EfficientVitStage(
                in_dim=pre_ed,
                out_dim=ed,
                key_dim=kd,
                downsample=do,
                num_heads=nh,
                attn_ratio=ar,
                resolution=resolution,
                window_resolution=wd,
                kernels=kernels,
                depth=dpth,
            )
            pre_ed = ed
            if do[0] == 'subsample' and i != 0:
                stride *= do[1]
            resolution = stage.resolution
            stages.append(stage)
            self.feature_info += [dict(num_chs=ed, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential()
        self.stages.add(stages)

        if global_pool == 'avg':
            self.global_pool = nn.SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        else:
            assert num_classes == 0
            self.global_pool = nn.identity()
        self.num_features = self.head_hidden_size = embed_dim[-1]
        self.head = NormLinear(
            self.num_features, num_classes, drop=self.drop_rate) if num_classes > 0 else nn.identity()

    def no_weight_decay(self):
        return ['attention_biases']

    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def get_classifier(self):
        return self.head.linear

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            if global_pool == 'avg':
                self.global_pool = nn.SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
            else:
                assert num_classes == 0
                self.global_pool = nn.identity()
        self.head = NormLinear(
            self.num_features, num_classes, drop=self.drop_rate) if num_classes > 0 else nn.identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        return x if pre_logits else self.head(x)

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def efficientvit_m0(**kwargs):
    return EfficientVitMsra(img_size=224,
                            embed_dim=[64, 128, 192],
                            depth=[1, 2, 3],
                            num_heads=[4, 4, 4],
                            window_size=[7, 7, 7],
                            kernels=[5, 5, 5, 5])


def efficientvit_m1(**kwargs):
    return EfficientVitMsra(img_size=224,
                            embed_dim=[128, 144, 192],
                            depth=[1, 2, 3],
                            num_heads=[2, 3, 3],
                            window_size=[7, 7, 7],
                            kernels=[7, 5, 3, 3])


def efficientvit_m2(**kwargs):
    return EfficientVitMsra(img_size=224,
                            embed_dim=[128, 192, 224],
                            depth=[1, 2, 3],
                            num_heads=[4, 3, 2],
                            window_size=[7, 7, 7],
                            kernels=[7, 5, 3, 3])


def efficientvit_m3(**kwargs):
    return EfficientVitMsra(img_size=224,
                            embed_dim=[128, 240, 320],
                            depth=[1, 2, 3],
                            num_heads=[4, 3, 4],
                            window_size=[7, 7, 7],
                            kernels=[5, 5, 5, 5])


def efficientvit_m4(**kwargs):
    return EfficientVitMsra(img_size=224,
                            embed_dim=[128, 256, 384],
                            depth=[1, 2, 3],
                            num_heads=[4, 4, 4],
                            window_size=[7, 7, 7],
                            kernels=[7, 5, 3, 3])


def efficientvit_m5(**kwargs):
    return EfficientVitMsra(img_size=224,
                            embed_dim=[192, 288, 384],
                            depth=[1, 3, 4],
                            num_heads=[3, 3, 4],
                            window_size=[7, 7, 7],
                            kernels=[7, 5, 3, 3])