""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2025 NoteDance
"""
import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from Note import nn

_logger = logging.getLogger(__name__)


class PatchEmbed:
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        self.patch_size = nn.to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = nn.Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = nn.Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.conv2d(embed_dim, input_size=in_chans, kernel_size=patch_size, strides=patch_size, use_bias=bias)
        self.pad = nn.zeropadding2d()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = nn.to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = nn.to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            new_proj = nn.conv2d(
                self.proj.out_channels,
                input_size=self.proj.in_channels,
                kernel_size=new_patch_size,
                strides=new_patch_size,
                use_bias=self.proj.bias is not None,
            )
            new_proj.weight.assign(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
            if self.proj.bias is not None:
                new_proj.bias.assign(self.proj.bias)
            self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[0] == 0, f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert W % self.patch_size[1] == 0, f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = self.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        x = self.proj(x)
        if self.flatten:
            x = tf.reshape(x, (B, -1, C))
        elif self.output_fmt != nn.Format.NCHW:
            x = nn.nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class PatchEmbedWithSize(PatchEmbed):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.img_size is not None:
            assert H % self.patch_size[0] == 0, f"Input image height ({H}) must be divisible by patch size ({self.patch_size[0]})."
            assert W % self.patch_size[1] == 0, f"Input image width ({W}) must be divisible by patch size ({self.patch_size[1]})."

        x = self.proj(x)
        feat_size = x.shape[1:3]
        if self.flatten:
            x = tf.reshape(x, (B, -1, C))
        elif self.output_fmt != nn.Format.NCHW:
            x = nn.nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x, feat_size


DTYPE_INTERMEDIATE = tf.float32


def _compute_resize_matrix(
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
    interpolation: str,
    antialias: bool,
    dtype = DTYPE_INTERMEDIATE
):
    """Computes the resize matrix basis vectors and interpolates them to new_size."""
    old_h, old_w = old_size
    new_h, new_w = new_size
    old_total = old_h * old_w
    new_total = new_h * new_w

    eye_matrix = tf.eye(old_total, dtype=dtype)
    basis_vectors_batch = tf.reshape(eye_matrix, (old_total, old_h, old_w, 1))
    resized_basis_vectors_batch = nn.interpolate(
        basis_vectors_batch,
        size=new_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False
    ) # Output shape: (old_total, 1, new_h, new_w)
    resize_matrix = resized_basis_vectors_batch.squeeze(1).permute(1, 2, 0).reshape(new_total, old_total)
    return resize_matrix # Shape: (new_total, old_total)


def _apply_resampling(
    patch_embed,
    pinv_matrix,
    new_size_tuple: Tuple[int, int],
    orig_dtype,
    intermediate_dtype = DTYPE_INTERMEDIATE
):
    """ Simplified resampling w/o vmap use.
    As proposed by https://github.com/stas-sl
    """
    old_h, old_w, c_in, c_out = patch_embed.shape
    patch_embed = tf.cast(tf.reshape(patch_embed, [-1, c_in, c_out]), intermediate_dtype)
    pinv_matrix = tf.cast(pinv_matrix, intermediate_dtype)
    resampled_patch_embed = tf.tensordot(pinv_matrix, patch_embed, axes=[[1], [0]])
    resampled_patch_embed = tf.cast(tf.reshape(resampled_patch_embed, (*new_size_tuple, c_out, c_in)), orig_dtype)
    return resampled_patch_embed


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """ Standalone function (computes matrix on each call). """
    assert len(patch_embed.shape) == 4, "Input tensor should be 4D (out_ch, in_ch, h, w)"
    assert len(new_size) == 2, "New shape should only be hw (height, width)"

    old_size_tuple: Tuple[int, int] = tuple(patch_embed.shape[-2:])
    new_size_tuple: Tuple[int, int] = tuple(new_size)

    if old_size_tuple == new_size_tuple:
        return patch_embed

    device = patch_embed.device
    orig_dtype = patch_embed.dtype

    resize_mat = _compute_resize_matrix(
        old_size_tuple, new_size_tuple, interpolation, antialias, device, DTYPE_INTERMEDIATE
    )
    pinv_matrix = tf.linalg.pinv(resize_mat)  # Calculates the pseudoinverse matrix used for resampling
    resampled_patch_embed = _apply_resampling(
        patch_embed, pinv_matrix, new_size_tuple, orig_dtype, DTYPE_INTERMEDIATE
    )
    return resampled_patch_embed