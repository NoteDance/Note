""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
import math
from typing import List, Optional


def rot(x):
    return tf.reshape(tf.stack([-x[..., 1::2], x[..., ::2]], -1), x.shape)


def apply_rot_embed(x, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * tf.broadcast_to(tf.expand_dims(cos_emb, axis=1), x.shape) + \
                rot(x) * tf.broadcast_to(tf.expand_dims(sin_emb, axis=1), x.shape)
    return x * cos_emb + rot(x) * sin_emb


def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
):
    if linear_bands:
        bands = tf.cast(tf.linspace(1.0, max_freq / 2, num_bands), 'float32')
    else:
        bands = 2 ** tf.cast(tf.linspace(0, math.log(max_freq, 2) - 1, num_bands), 'float32')
    return bands * tf.constant(3.141592653589793)


def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
):
    exp = tf.cast(tf.range(0, num_bands, step, dtype=tf.int64), 'float32') / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def ndgrid(*tensors):
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    """
    try:
        return tf.meshgrid(*tensors, indexing='ij')
    except TypeError:
        return tf.meshgrid(*tensors)


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands = None,
        num_bands: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype = tf.float32,
):
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
            )
    else:
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [tf.cast(tf.linspace(-1., 1., s), 'float32') for s in feat_shape]
    else:
        t = [tf.cast(tf.range(s, dtype=tf.int64), tf.float32) for s in feat_shape]

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = tf.stack(ndgrid(t), axis=-1)
    grid = tf.expand_dims(grid, axis=-1)
    pos = grid * bands

    pos_sin, pos_cos = tf.cast(tf.math.sin(pos), dtype), tf.cast(tf.math.cos(pos), dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


def build_rotary_pos_embed(
        feat_shape: List[int],
        bands = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype = tf.float32,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        dtype=dtype,
    )
    num_spatial_dim = 1
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = tf.reshape(sin_emb, [num_spatial_dim, -1])
    cos_emb = tf.reshape(cos_emb, [num_spatial_dim, -1])
    sin_emb = tf.repeat(sin_emb, repeats=2, axis=-1)
    cos_emb = tf.repeat(cos_emb, repeats=2, axis=-1)
    return sin_emb, cos_emb


class RotaryEmbedding:
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
    ):
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
            self.bands = bands
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            emb_sin, emb_cos = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )
            self.bands = None
            self.pos_embed_sin = emb_sin
            self.pos_embed_cos = emb_cos

    def get_embed(self, shape: Optional[List[int]] = None):
        if self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            assert shape is not None
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
            )
        else:
            return self.pos_embed_sin, self.pos_embed_cos

    def __call__(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)