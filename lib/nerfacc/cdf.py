"""
Copyright (c) 2023 Shaofei Wang, ETH Zurich.
"""

from typing import Tuple

from torch import Tensor

import lib.nerfacc.cuda as _C


def ray_resampling(
    packed_info: Tensor,
    t_starts: Tensor,
    t_ends: Tensor,
    weights: Tensor,
    n_samples: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resample a set of rays based on the CDF of the weights. This is the modified version of
    nerfacc-0.3.0's ray_resampling.  It samples midpoints instead of starts/ends of the
    frustum-shape samples. Further, it does not normalize weights, but adds a background interval at
    this end of the CDF with a weight `1.0 - weights_sum`, this enables samples to be assigned to
    the background when the foreground density is low.

    Args:
        packed_info (Tensor): Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples_in, 1).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples_in, 1).
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples_in,).
        n_samples (int): Number of samples per ray to resample.

    Returns:
        resampled_packed_info (Tensor): Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        resampled_ts (Tensor): midpoints of the resampled frustum-shape samples. Tensor with shape \
            (n_samples, 1).
        resampled_offsets (Tensor): Stores the offsets of the resampled midpoints wrt. the starts \
            of the sampled intervals. Tensor with shape (n_samples, 1).
        resampled_indices (Tensor): Stores the indices of sampled intervals for each resampled \
            midpoint. Tensor with shape (n_samples,).
        resampled_fg_counts (Tensor): Stores the number of foreground samples for each interval. Tensor \
            with shape (n_samples_in,).
        resampled_bg_counts (Tensor): Stores the number of background samples for each ray. Tensor \
            with shape (n_rays,).
    """
    assert (n_samples > 1)  # the CUDA kernel does not handle n_samples == 1
    (
        resampled_packed_info,
        resampled_ts,
        resampled_offsets,
        resampled_indices,
        resampled_fg_counts,
        resampled_bg_counts,
    ) = _C.ray_resampling(
        packed_info.contiguous(),
        t_starts.contiguous(),
        t_ends.contiguous(),
        weights.contiguous(),
        n_samples,
    )
    return resampled_packed_info, resampled_ts, resampled_offsets, resampled_indices, resampled_fg_counts, resampled_bg_counts
