from typing import Callable, Optional, Tuple

import torch

import nerfacc.cuda as _C

from nerfacc.contraction import ContractionType
from nerfacc.grid import Grid
from nerfacc.intersection import ray_aabb_intersect
from nerfacc.vol_rendering import render_weight_from_alpha, accumulate_along_rays

"""
This code is adapted from NerfAcc
"""


@torch.no_grad()
def transmittance(
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_min: Optional[torch.Tensor] = None,
    t_max: Optional[torch.Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[torch.Tensor] = None,
    # binarized grid for skipping empty space
    grid: Optional[Grid] = None,
    # sigma/alpha function for evaluating transmittance
    sigma_fn: Optional[Callable] = None,
    alpha_fn: Optional[Callable] = None,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor,]:
    """Compute the transmittance of input rays."""
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if alpha_fn is not None and sigma_fn is not None:
        raise ValueError("Only one of `alpha_fn` and `sigma_fn` should be provided.")

    # logic for t_min and t_max:
    # 1. if t_min and t_max are given, use them with highest priority.
    # 2. if t_min and t_max are not given, but scene_aabb is given, use
    # ray_aabb_intersect to compute t_min and t_max.
    # 3. if t_min and t_max are not given, and scene_aabb is not given,
    # set t_min to 0.0, and t_max to 1e10. (the case of unbounded scene)
    # 4. always clip t_min with near_plane and t_max with far_plane if given.
    if t_min is None or t_max is None:
        if scene_aabb is not None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, scene_aabb)
        else:
            t_min = torch.zeros_like(rays_o[..., 0])
            t_max = torch.ones_like(rays_o[..., 0]) * 1e10
    if near_plane is not None:
        t_min = torch.clamp(t_min, min=near_plane)
    if far_plane is not None:
        t_max = torch.clamp(t_max, max=far_plane)

    # stratified sampling: prevent overfitting during training
    if stratified:
        t_min = t_min + torch.rand_like(t_min) * render_step_size

    # use grid for skipping if given
    if grid is not None:
        grid_roi_aabb = grid.roi_aabb
        grid_binary = grid.binary
        contraction_type = grid.contraction_type.to_cpp_version()
    else:
        grid_roi_aabb = torch.tensor(
            [-1e10, -1e10, -1e10, 1e10, 1e10, 1e10],
            dtype=torch.float32,
            device=rays_o.device,
        )
        grid_binary = torch.ones(
            [1, 1, 1], dtype=torch.bool, device=rays_o.device
        )
        contraction_type = ContractionType.AABB.to_cpp_version()

    packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # coontraction and grid
        grid_roi_aabb.contiguous(),
        grid_binary.contiguous(),
        contraction_type,
        # sampling
        render_step_size,
        cone_angle,
    )

    # Find canonical correspondences
    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]
    midpoints = (t_starts + t_ends) / 2.0
    positions = t_origins + t_dirs * midpoints
    dists = t_ends - t_starts

    # Change the signature of sigma_fn/alpha_fn,
    # directly input canonical coordinates
    # skip invisible space
    if sigma_fn is not None or alpha_fn is not None:
        # Query sigma without gradients
        if sigma_fn is not None:
            sigmas = sigma_fn(rays_d, positions, dists, ray_indices)
            assert (
                sigmas.shape == t_starts.shape
            ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
            alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        elif alpha_fn is not None:
            alphas = alpha_fn(rays_d, positions, dists, ray_indices)
            assert (
                alphas.shape == t_starts.shape
            ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)

        weights = render_weight_from_alpha(
            alphas, packed_info=None, ray_indices=ray_indices, n_rays=rays_o.shape[0]
        )

        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=rays_o.shape[0])

    return 1.0 - opacity
