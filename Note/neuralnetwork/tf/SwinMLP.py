# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2024 NoteDance
# Licensed under The MIT License [see LICENSE for details]
# Written by NoteDance
# --------------------------------------------------------

import tensorflow as tf
from Note import nn
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class Mlp:
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.dense(hidden_features, in_features)
        self.act = act_layer
        self.fc2 = nn.dense(out_features, hidden_features)
        self.drop = nn.dropout(drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (B, H, W, -1))
    return x


class SwinMLPBlock:
    r""" Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer: Activation layer. Default: tf.nn.gelu
        norm_layer: Normalization layer.  Default: nn.layer_norm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=tf.nn.gelu, norm_layer=nn.layer_norm):
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.group_conv1d(self.num_heads * self.window_size ** 2,
                                     input_size=self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     num_groups=self.num_heads)

        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))

        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = tf.pad(x, [[0, 0,], [P_l, P_r,], [P_t, P_b], [0, 0,]], "CONSTANT", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))  # nW*B, window_size*window_size, C
        
        # Window/Shifted-Window Spatial MLP
        x_windows_heads = tf.reshape(x_windows, (-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads))
        x_windows_heads = tf.transpose(x_windows_heads, (0, 2, 1, 3))  # nW*B, nH, window_size*window_size, C//nH
        x_windows_heads = tf.reshape(x_windows_heads, (-1, C // self.num_heads, 
                                                       self.num_heads * self.window_size * self.window_size))
        spatial_mlp_windows = tf.transpose(self.spatial_mlp(x_windows_heads), (0, 2, 1))  # nW*B, nH*window_size*window_size, C//nH
        spatial_mlp_windows = tf.transpose(tf.reshape(spatial_mlp_windows, (-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads)), (0, 2, 1, 3))
        spatial_mlp_windows = tf.reshape(spatial_mlp_windows, (-1, self.window_size * self.window_size, C))

        # merge windows
        spatial_mlp_windows = tf.reshape(spatial_mlp_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :]
        else:
            x = shifted_x
        x = tf.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W

        # Window/Shifted-Window Spatial MLP
        if self.shift_size > 0:
            nW = (H / self.window_size + 1) * (W / self.window_size + 1)
        else:
            nW = H * W / self.window_size / self.window_size
        flops += nW * self.dim * (self.window_size * self.window_size) * (self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging:
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer: Normalization layer.  Default: nn.layer_norm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.layer_norm):
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.dense(2 * dim, 4 * dim, use_bias=False)
        self.norm = norm_layer(4 * dim)

    def __call__(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, (B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = tf.reshape(x, (B, -1, 4 * C))  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer:
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer: Normalization layer. Default: nn.layer_norm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.layer_norm, downsample=None):

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = [
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, tf.Tensor) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed:
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer: Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.conv2d(embed_dim, input_size=in_chans, kernel_size=patch_size, strides=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def __call__(self, x):
        B, H, W, C = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H * W, C))  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinMLP(nn.Model):
    r""" Swin MLP

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer: Normalization layer. Default: nn.layer_norm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.layer_norm, ape=False, patch_norm=True,
                 **kwargs):
        super().__init__()
        nn.Model.add()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.initializer_((1, num_patches, embed_dim), ['truncated_normal', .02], name='absolute_pos_embed')

        self.pos_drop = nn.dropout(drop_rate)

        # stochastic depth
        dpr = tf.linspace(0., drop_path_rate, sum(depths))  # stochastic depth decay rule

        # build layers
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.adaptive_avg_pooling1d(1)
        self.head = nn.dense(num_classes, self.num_features) if num_classes > 0 else nn.identity()

        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', 0.2]))
        elif isinstance(l, nn.group_conv1d):
            for weight in l.weight:
                weight.assign(nn.initializer(weight.shape, ['truncated_normal', 0.2]))

    def no_weight_decay(self):
        return ['absolute_pos_embed']

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x) # B 1 C
        
        x = tf.reshape(x, (x.shape[0],-1))
        return x
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        self.flag=flag
        if flag==0:
            self.param_=self.param.copy()
            self.head_=self.head
            self.head=nn.dense(classes,self.num_features)
            param.extend(self.head.param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.head.param):]
            self.param_.extend(self.head.param)
            self.param=self.param_
        else:
            self.head,self.head_=self.head_,self.head
            del self.param_[-len(self.head.param):]
            self.param_.extend(self.head.param)
            self.param=self.param_
        return

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops