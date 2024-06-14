""" Position Embedding Utilities

Hacked together by / Copyright 2024 NoteDance
"""
import logging
import math
from typing import List, Optional

import tensorflow as tf
from Note import nn

_logger = logging.getLogger(__name__)


def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = tf.cast(posemb, 'float32')  # interpolate needs float32
    posemb = tf.reshape(posemb, (1, old_size[0], old_size[1], -1))
    posemb = nn.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = tf.reshape(posemb, (1, -1, embed_dim))
    posemb = tf.cast(posemb, orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = tf.concat([posemb_prefix, posemb], axis=1)

    if verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def resample_abs_pos_embed_nhwc(
        posemb,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    if new_size[0] == posemb.shape[-3] and new_size[1] == posemb.shape[-2]:
        return posemb

    orig_dtype = posemb.dtype
    posemb = tf.cast(posemb, 'float32')
    posemb = nn.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = tf.cast(posemb, orig_dtype)

    if verbose:
        _logger.info(f'Resized position embedding: {posemb.shape[-3:-1]} to {new_size}.')

    return posemb