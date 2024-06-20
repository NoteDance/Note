""" EfficientViT (by MIT Song Han's Lab)

Paper: `Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition`
    - https://arxiv.org/abs/2205.14756

Adapted from official impl at https://github.com/mit-han-lab/efficientvit
"""

__all__ = ['EfficientVit', 'EfficientVitLarge']
from typing import List, Optional
from functools import partial

import tensorflow as tf
from Note import nn


def val2list(x: list or tuple or any, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1):
    # repeat elements if necessary
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class ConvNormAct:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        dropout=0.,
        norm_layer=nn.batch_norm,
        act_layer=tf.nn.relu,
    ):
        self.dropout = nn.dropout(dropout)
        padding = get_padding(kernel_size, stride, dilation)
        self.conv = nn.conv2d(
            out_channels,
            kernel_size,
            in_channels,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=bias,
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.identity()
        self.act = act_layer if act_layer is not None else nn.identity()

    def __call__(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DSConv:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm_layer=(nn.batch_norm, nn.batch_norm),
        act_layer=(tf.nn.relu6, None),
    ):
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.depth_conv = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            in_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def __call__(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ConvBlock:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm_layer=(nn.batch_norm, nn.batch_norm),
        act_layer=(tf.nn.relu6, None),
    ):
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.conv2 = ConvNormAct(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MBConv:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm_layer=(nn.batch_norm, nn.batch_norm, nn.batch_norm),
        act_layer=(tf.nn.relu6, tf.nn.relu6, None),
    ):
        use_bias = val2tuple(use_bias, 3)
        norm_layer = val2tuple(norm_layer, 3)
        act_layer = val2tuple(act_layer, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            bias=use_bias[2],
        )

    def __call__(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm_layer=(nn.batch_norm, nn.batch_norm),
        act_layer=(tf.nn.relu6, None),
    ):
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def __call__(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class LiteMLA:
    """Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm_layer=(None, nn.batch_norm),
        act_layer=(None, None),
        kernel_func=tf.nn.relu,
        scales=(5,),
        eps=1e-5,
    ):
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.dim = dim
        self.qkv = ConvNormAct(
            in_channels,
            3 * total_dim,
            1,
            bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        self.aggreg = []
        for scale in scales:
            layers = nn.Sequential()
            layers.add(nn.conv2d(
                            3 * total_dim,
                            scale,
                            3 * total_dim,
                            padding=get_same_padding(scale),
                            groups=3 * total_dim,
                            use_bias=use_bias[0],
                            ))
            layers.add(nn.conv2d(3 * total_dim, 1, 3 * total_dim, groups=3 * heads, use_bias=use_bias[0]))
            self.aggreg.append(layers)
        self.kernel_func = kernel_func

        self.proj = ConvNormAct(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def _attn(self, q, k, v):
        dtype = v.dtype
        q, k, v = tf.cast(q, 'float32'), tf.cast(k, 'float32'), tf.cast(v, 'float32')
        kv = tf.matmul(tf.transpose(k, (0, 1, 3, 2)), v)
        out = tf.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        return tf.cast(out, dtype)

    def __call__(self, x):
        B, H, W, _ = x.shape

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = tf.concat(multi_scale_qkv, axis=-1)
        multi_scale_qkv = tf.reshape(multi_scale_qkv, (B, -1, H * W, 3 * self.dim))
        q, k, v = tf.split(multi_scale_qkv, num_or_size_splits=3, axis=-1)

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = tf.pad(v, [[0, 0], [0, 0], [0, 0], [0, 1]], "CONSTANT", constant_values=1.0)

        out = self._attn(q, k, v)

        # final projection
        out = tf.reshape(out, (B, H, W, -1))
        out = self.proj(out)
        return out


class EfficientVitBlock:
    def __init__(
        self,
        in_channels,
        heads_ratio=1.0,
        head_dim=32,
        expand_ratio=4,
        norm_layer=nn.batch_norm,
        act_layer=nn.activation_dict['hardswish'],
    ):
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=head_dim,
                norm_layer=(None, norm_layer),
            ),
            nn.identity(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            ),
            nn.identity(),
        )

    def __call__(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock:
    def __init__(
        self,
        main,
        shortcut = None,
        pre_norm = None,
    ):
        self.pre_norm = pre_norm if pre_norm is not None else nn.identity()
        self.main = main
        self.shortcut = shortcut

    def __call__(self, x):
        res = self.main(self.pre_norm(x))
        if self.shortcut is not None:
            res = res + self.shortcut(x)
        return res


def build_local_block(        
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm_layer: str,
        act_layer: str,
        fewer_norm: bool = False,
        block_type: str = "default",
):
    assert block_type in ["default", "large", "fused"]
    if expand_ratio == 1:
        if block_type == "default":
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
        else:
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    else:
        if block_type == "default":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm_layer=(None, None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, act_layer, None),
            )
        else:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    return block


class Stem(nn.Sequential):
    def __init__(self, in_chs, out_chs, depth, norm_layer, act_layer, block_type='default'):
        super().__init__()
        self.stride = 2

        self.add(
            ConvNormAct(
                in_chs, out_chs,
                kernel_size=3, stride=2, norm_layer=norm_layer, act_layer=act_layer,
            )
        )
        stem_block = 0
        for _ in range(depth):
            self.add(ResidualBlock(
                build_local_block(
                    in_channels=out_chs,
                    out_channels=out_chs,
                    stride=1,
                    expand_ratio=1,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    block_type=block_type,
                ),
                nn.identity(),
            ))
            stem_block += 1


class EfficientVitStage:
    def __init__(
            self,
            in_chs,
            out_chs,
            depth,
            norm_layer,
            act_layer,
            expand_ratio,
            head_dim,
            vit_stage=False,
    ):
        blocks = [ResidualBlock(
            build_local_block(
                in_channels=in_chs,
                out_channels=out_chs,
                stride=2,
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                fewer_norm=vit_stage,
            ),
            None,
        )]
        in_chs = out_chs

        if vit_stage:
            # for stage 3, 4
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=expand_ratio,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            # for stage 1, 2
            for i in range(1, depth):
                blocks.append(ResidualBlock(
                    build_local_block(
                        in_channels=in_chs,
                        out_channels=out_chs,
                        stride=1,
                        expand_ratio=expand_ratio,
                        norm_layer=norm_layer,
                        act_layer=act_layer
                    ),
                    nn.identity(),
                ))

        self.blocks = nn.Sequential()
        self.blocks.add(blocks)

    def __call__(self, x):
        return self.blocks(x)


class EfficientVitLargeStage:
    def __init__(
            self,
            in_chs,
            out_chs,
            depth,
            norm_layer,
            act_layer,
            head_dim,
            vit_stage=False,
            fewer_norm=False,
    ):
        blocks = [ResidualBlock(
            build_local_block(
                in_channels=in_chs,
                out_channels=out_chs,
                stride=2,
                expand_ratio=24 if vit_stage else 16,
                norm_layer=norm_layer,
                act_layer=act_layer,
                fewer_norm=vit_stage or fewer_norm,
                block_type='default' if fewer_norm else 'fused',
            ),
            None,
        )]
        in_chs = out_chs

        if vit_stage:
            # for stage 4
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=6,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            # for stage 1, 2, 3
            for i in range(depth):
                blocks.append(ResidualBlock(
                    build_local_block(
                        in_channels=in_chs,
                        out_channels=out_chs,
                        stride=1,
                        expand_ratio=4,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        fewer_norm=fewer_norm,
                        block_type='default' if fewer_norm else 'fused',
                    ),
                    nn.identity(),
                ))

        self.blocks = nn.Sequential()
        self.blocks.add(blocks)

    def __call__(self, x):
        return self.blocks(x)


class ClassifierHead:
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        num_classes: int = 1000,
        dropout: float = 0.,
        norm_layer=nn.batch_norm,
        act_layer=nn.activation_dict['hardswish'],
        pool_type: str = 'avg',
        norm_eps: float = 1e-5,
    ):
        self.widths = widths
        self.num_features = widths[-1]

        assert pool_type, 'Cannot disable pooling'
        self.in_conv = ConvNormAct(in_channels, widths[0], 1, norm_layer=norm_layer, act_layer=act_layer)
        self.global_pool = nn.SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        self.classifier = nn.Sequential()
        self.classifier.add(nn.dense(widths[1], widths[0], use_bias=False))
        self.classifier.add(nn.layer_norm(widths[1], epsilon=norm_eps))
        self.classifier.add(act_layer if act_layer is not None else nn.identity())
        self.classifier.add(nn.dropout(dropout))
        self.classifier.add(nn.dense(num_classes, widths[1], use_bias=True) if num_classes > 0 else nn.identity())

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            assert pool_type, 'Cannot disable pooling'
            self.global_pool = nn.SelectAdaptivePool2d(pool_type=pool_type, flatten=True,)
        if num_classes > 0:
            self.classifier[-1] = nn.dense(num_classes, self.num_features, use_bias=True)
        else:
            self.classifier[-1] = nn.identity()

    def __call__(self, x, pre_logits: bool = False):
        x = self.in_conv(x)
        x = self.global_pool(x)
        if pre_logits:
            # cannot slice or iterate with torchscript so, this
            x = self.classifier.layer[0](x)
            x = self.classifier.layer[1](x)
            x = self.classifier.layer[2](x)
            x = self.classifier.layer[3](x)
        else:
            x = self.classifier(x)
        return x


class EfficientVit(nn.Model):
    def __init__(
        self,
        in_chans=3,
        widths=(),
        depths=(),
        head_dim=32,
        expand_ratio=4,
        norm_layer=nn.batch_norm,
        act_layer=nn.activation_dict['hardswish'],
        global_pool='avg',
        head_widths=(),
        drop_rate=0.0,
        num_classes=1000,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.global_pool = global_pool
        self.num_classes = num_classes

        # input stem
        self.stem = Stem(in_chans, widths[0], depths[0], norm_layer, act_layer)
        stride = self.stem.stride

        # stages
        self.feature_info = []
        self.stages = nn.Sequential()
        in_channels = widths[0]
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            self.stages.add(EfficientVitStage(
                in_channels,
                w,
                depth=d,
                norm_layer=norm_layer,
                act_layer=act_layer,
                expand_ratio=expand_ratio,
                head_dim=head_dim,
                vit_stage=i >= 2,
            ))
            stride *= 2
            in_channels = w
            self.feature_info += [dict(num_chs=in_channels, reduction=stride, module=f'stages.{i}')]

        self.num_features = in_channels
        self.head = ClassifierHead(
            self.num_features,
            widths=head_widths,
            num_classes=num_classes,
            dropout=drop_rate,
            pool_type=self.global_pool,
        )
        self.head_hidden_size = self.head.num_features

    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def get_classifier(self):
        return self.head.classifier[-1]

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class EfficientVitLarge(nn.Model):
    def __init__(
        self,
        in_chans=3,
        widths=(),
        depths=(),
        head_dim=32,
        norm_layer=nn.batch_norm,
        act_layer=partial(tf.nn.gelu, approximate='tanh'),
        global_pool='avg',
        head_widths=(),
        drop_rate=0.0,
        num_classes=1000,
        norm_eps=1e-7,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.norm_eps = norm_eps
        norm_layer = partial(norm_layer, epsilon=self.norm_eps)

        # input stem
        self.stem = Stem(in_chans, widths[0], depths[0], norm_layer, act_layer, block_type='large')
        stride = self.stem.stride

        # stages
        self.feature_info = []
        self.stages = nn.Sequential()
        in_channels = widths[0]
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            self.stages.add(EfficientVitLargeStage(
                in_channels,
                w,
                depth=d,
                norm_layer=norm_layer,
                act_layer=act_layer,
                head_dim=head_dim,
                vit_stage=i >= 3,
                fewer_norm=i >= 2,
            ))
            stride *= 2
            in_channels = w
            self.feature_info += [dict(num_chs=in_channels, reduction=stride, module=f'stages.{i}')]

        self.num_features = in_channels
        self.head = ClassifierHead(
            self.num_features,
            widths=head_widths,
            num_classes=num_classes,
            dropout=drop_rate,
            pool_type=self.global_pool,
            act_layer=act_layer,
            norm_eps=self.norm_eps,
        )
        self.head_hidden_size = self.head.num_features

    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def get_classifier(self):
        return self.head.classifier[-1]

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def efficientvit_b0(**kwargs):
    return EfficientVit(widths=(8, 16, 32, 64, 128), depths=(1, 2, 2, 2, 2), head_dim=16, head_widths=(1024, 1280))


def efficientvit_b1(**kwargs):
    return EfficientVit(widths=(16, 32, 64, 128, 256), depths=(1, 2, 3, 3, 4), head_dim=16, head_widths=(1536, 1600))


def efficientvit_b2(**kwargs):
    return EfficientVit(widths=(24, 48, 96, 192, 384), depths=(1, 3, 4, 4, 6), head_dim=32, head_widths=(2304, 2560))


def efficientvit_b3(**kwargs):
    return EfficientVit(widths=(32, 64, 128, 256, 512), depths=(1, 4, 6, 6, 9), head_dim=32, head_widths=(2304, 2560))


def efficientvit_l1(**kwargs):
    return EfficientVitLarge(widths=(32, 64, 128, 256, 512), depths=(1, 1, 1, 6, 6), head_dim=32, head_widths=(3072, 3200))


def efficientvit_l2(**kwargs):
    return EfficientVitLarge(widths=(32, 64, 128, 256, 512), depths=(1, 2, 2, 8, 8), head_dim=32, head_widths=(3072, 3200))


def efficientvit_l3(**kwargs):
    return EfficientVitLarge(widths=(64, 128, 256, 512, 1024), depths=(1, 2, 2, 8, 8), head_dim=32, head_widths=(6144, 6400))