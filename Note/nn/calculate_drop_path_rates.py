from typing import Union, List
import tensorflow as tf


def calculate_drop_path_rates(
        drop_path_rate: float,
        depths: Union[int, List[int]],
        stagewise: bool = False,
) -> Union[List[float], List[List[float]]]:
    """Generate drop path rates for stochastic depth.

    This function handles two common patterns for drop path rate scheduling:
    1. Per-block: Linear increase from 0 to drop_path_rate across all blocks
    2. Stage-wise: Linear increase across stages, with same rate within each stage

    Args:
        drop_path_rate: Maximum drop path rate (at the end).
        depths: Either a single int for total depth (per-block mode) or
                list of ints for depths per stage (stage-wise mode).
        stagewise: If True, use stage-wise pattern. If False, use per-block pattern.
                   When depths is a list, stagewise defaults to True.

    Returns:
        For per-block mode: List of drop rates, one per block.
        For stage-wise mode: List of lists, drop rates per stage.
    """
    if isinstance(depths, int):
        # Single depth value - per-block pattern
        if stagewise:
            raise ValueError("stagewise=True requires depths to be a list of stage depths")
        dpr = list(tf.linspace(0.0, drop_path_rate, depths).numpy())
        return dpr
    else:
        # List of depths - can be either pattern
        total_depth = sum(depths)
        lin = tf.linspace(0.0, drop_path_rate, total_depth)
        if stagewise:
            # Stage-wise pattern: same drop rate within each stage
            split = tf.split(lin, depths)
            dpr = [list(s.numpy()) for s in split]
            return dpr
        else:
            # Per-block pattern across all stages
            dpr = list(lin.numpy())
            return dpr