"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from nerfacc.volrend import (
    render_weight_from_density,
    render_weight_from_alpha,
    accumulate_along_rays,
)

from models.utils import chunk_batch

def secondary_rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    alpha_fn: Optional[Callable] = None,
    # chunk options
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    # Query sigma/alpha and color with gradients

    if alpha_fn is not None:
        if t_starts.shape[0] != 0:
            if chunk_size is None:
                alphas = alpha_fn(t_starts, t_ends, ray_indices)
            else:
                alphas = chunk_batch(
                    alpha_fn,
                    chunk_size,
                    False,
                    t_starts,
                    t_ends,
                    ray_indices,
                )
        else:
            alphas = torch.empty((0,), device=t_starts.device)
    
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "alphas": alphas,
        }

    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )


    return opacities, depths, extras

def rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    # chunk options
    chunk_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        if t_starts.shape[0] != 0:
            if chunk_size is None:
                rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
            else:
                rgbs, sigmas = chunk_batch(
                    rgb_sigma_fn,
                    chunk_size,
                    False,
                    t_starts,
                    t_ends,
                    ray_indices,
                )
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            if chunk_size is None:
                rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
            else:
                rgbs, alphas = chunk_batch(
                    rgb_alpha_fn,
                    chunk_size,
                    False,
                    t_starts,
                    t_ends,
                    ray_indices,
                )
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
        }

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, extras


def rendering_with_normals(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    ray_march_thres: float = 0.0,
    render_bkgd: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        if t_starts.shape[0] != 0:
            rgbs, normals, derived_normals, sigmas = rgb_sigma_fn(
                t_starts, t_ends, ray_indices
            )
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            normals = torch.empty((0, 3), device=t_starts.device)
            derived_normals = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == 3, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            derived_normals.shape[-1] == 3
        ), "derived_normals must have 3 channels, got {}".format(derived_normals.shape)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        mask = weights > ray_march_thres
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
            "normals": normals,
            "derived_normals": derived_normals,
            "mask": mask,
        }
        app_weights = weights[mask]
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            rgbs, normals, derived_normals, alphas = rgb_alpha_fn(
                t_starts, t_ends, ray_indices
            )
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            normals = torch.empty((0, 3), device=t_starts.device)
            derived_normals = torch.empty((0, 3), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == 3, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            derived_normals.shape[-1] == 3
        ), "derived_normals must have 3 channels, got {}".format(derived_normals.shape)
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        mask = weights > ray_march_thres
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
            "normals": normals,
            "derived_normals": derived_normals,
            "mask": mask,
        }
        app_weights = weights[mask]

    # Rendering: accumulate rgbs, normals, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        app_weights, values=rgbs[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    normals = accumulate_along_rays(
        app_weights, values=normals[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    # depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)
        normals = normals + render_bkgd * (1 - opacities) * torch.tensor(
            [0.0, 0.0, 1.0], device=normals.device
        )  # Background normal

    return colors, normals, opacities, depths, extras


def rendering_with_normals_mats(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    ray_march_thres: float = 0.0,
    render_bkgd: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        if t_starts.shape[0] != 0:
            (
                rgbs,
                normals,
                materials,
                materials_jitter,
                derived_normals,
                sigmas,
            ) = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            normals = torch.empty((0, 3), device=t_starts.device)
            materials = torch.empty((0, 5), device=t_starts.device)
            materials_jitter = torch.empty((0, 5), device=t_starts.device)
            derived_normals = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == 3, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            materials.shape[-1] == 5
        ), "materials must have 5 channels, got {}".format(materials.shape)
        assert (
            materials_jitter.shape[-1] == 5
        ), "materials_jitter must have 5 channels, got {}".format(
            materials_jitter.shape
        )
        assert (
            derived_normals.shape[-1] == 3
        ), "derived_normals must have 3 channels, got {}".format(derived_normals.shape)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        albedo, roughness, metallic = (
            materials[..., :3],
            materials[..., 3:4] * 0.9 + 0.09,
            materials[..., 4:5],
        )
        albedo_jitter, roughness_jitter, metallic_jitter = (
            materials_jitter[..., :3],
            materials_jitter[..., 3:4] * 0.9 + 0.09,
            materials_jitter[..., 4:5],
        )
        mask = weights > ray_march_thres
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
            "normals": normals,
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "albedo_jitter": albedo_jitter,
            "roughness_jitter": roughness_jitter,
            "metallic_jitter": metallic_jitter,
            "derived_normals": derived_normals,
            "mask": mask,
        }
        app_weights = weights[mask]
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            (
                rgbs,
                normals,
                materials,
                materials_jitter,
                derived_normals,
                alphas,
            ) = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            normals = torch.empty((0, 3), device=t_starts.device)
            materials = torch.empty((0, 5), device=t_starts.device)
            materials_jitter = torch.empty((0, 5), device=t_starts.device)
            derived_normals = torch.empty((0, 3), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == 3, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            materials.shape[-1] == 5
        ), "materials must have 5 channels, got {}".format(materials.shape)
        assert (
            materials_jitter.shape[-1] == 5
        ), "materials_jitter must have 5 channels, got {}".format(
            materials_jitter.shape
        )
        assert (
            derived_normals.shape[-1] == 3
        ), "derived_normals must have 3 channels, got {}".format(derived_normals.shape)
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        albedo, roughness, metallic = (
            materials[..., :3],
            materials[..., 3:4] * 0.9 + 0.09,
            materials[..., 4:5],
        )
        albedo_jitter, roughness_jitter, metallic_jitter = (
            materials_jitter[..., :3],
            materials_jitter[..., 3:4] * 0.9 + 0.09,
            materials_jitter[..., 4:5],
        )
        mask = weights > ray_march_thres
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
            "normals": normals,
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "albedo_jitter": albedo_jitter,
            "roughness_jitter": roughness_jitter,
            "metallic_jitter": metallic_jitter,
            "derived_normals": derived_normals,
            "mask": mask,
        }
        app_weights = weights[mask]

    # Rendering: accumulate rgbs, normals, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        app_weights, values=rgbs[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    normals = accumulate_along_rays(
        app_weights, values=normals[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    albedo = accumulate_along_rays(
        app_weights, values=albedo[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    roughness = accumulate_along_rays(
        app_weights, values=roughness[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    metallic = accumulate_along_rays(
        app_weights, values=metallic[mask], ray_indices=ray_indices[mask], n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    # depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)
        normals = normals + render_bkgd * (1 - opacities) * torch.tensor(
            [0.0, 0.0, 1.0], device=normals.device
        )  # Background normal
        albedo = albedo + render_bkgd * (1.0 - opacities)
        roughness = roughness + render_bkgd * (1.0 - opacities)

    return colors, normals, albedo, roughness, metallic, opacities, depths, extras


def rendering_with_normals_sdf(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    has_laplace: bool = False,
    color_dim = 3,
    normal_dim = 3,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        raise NotImplementedError("rgb_sigma_fn is not implemented yet.")
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            if has_laplace:

                rgbs, normals, alphas, sdf, sdf_grad, sdf_laplace = rgb_alpha_fn(
                    t_starts, t_ends, ray_indices
                )
            else:
                rgbs, normals, alphas, sdf, sdf_grad = rgb_alpha_fn(
                    t_starts, t_ends, ray_indices
                )
        else:
            rgbs = torch.empty((0, color_dim), device=t_starts.device)
            normals = torch.empty((0, normal_dim), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
            sdf = torch.empty((0,), device=t_starts.device)
            sdf_grad = torch.empty((0, 3), device=t_starts.device)
            sdf_laplace = torch.empty((0, 3), device=t_starts.device)
        assert rgbs.shape[-1] == color_dim, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == normal_dim, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        assert (
            sdf.shape == t_starts.shape
        ), "sdf must have shape of (N,)! Got {}".format(sdf.shape)
        assert (
            sdf_grad.shape[-1] == 3
        ), "sdf_grad must have 3 channels, got {}".format(sdf_grad.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
            "normals": normals,
            "sdf": sdf,
            "sdf_grad": sdf_grad,
        }
        if has_laplace:
            extras.update({
                "sdf_laplace": sdf_laplace
            })

    # Rendering: accumulate rgbs, normals, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    normals = accumulate_along_rays(
        weights, values=normals, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    # depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)
        normals = normals + render_bkgd * (1 - opacities) * torch.tensor(
            [0.0, 0.0, 1.0], device=normals.device
        )  # Background normal

    return colors, normals, opacities, depths, extras


def rendering_with_normals_mats_sdf(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    material_dim = 5,
    laplacian=False
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).
        material_dim: dimension of material. Must be either 5 or 7.  5 = 3-albedo + 1-roughness +
            1-metallic. 7 = 3-albedo + 1-roughness + 3-metallic - in this case metallic means
            specular albedo.

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        raise NotImplementedError("rgb_sigma_fn is not implemented yet.")
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            if laplacian:
                (
                    rgbs,
                    normals,
                    materials,
                    materials_jitter,
                    alphas,
                    sigmas,
                    sdf,
                    sdf_grad,
                    t_dirs,
                    sdf_laplace
                ) = rgb_alpha_fn(t_starts, t_ends, ray_indices)
            else:
                (
                    rgbs,
                    normals,
                    materials,
                    materials_jitter,
                    alphas,
                    sigmas,
                    sdf,
                    sdf_grad,
                ) = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            normals = torch.empty((0, 3), device=t_starts.device)
            materials = torch.empty((0, material_dim), device=t_starts.device)
            materials_jitter = torch.empty((0, material_dim), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
            sdf = torch.empty((0,), device=t_starts.device)
            sdf_grad = torch.empty((0, 3), device=t_starts.device)
            sdf_laplace = torch.empty((0,), device=t_starts.device)
            t_dirs = torch.empty((0, 3), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == 3, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            materials.shape[-1] == 5 or materials.shape[-1] == 7
        ), "materials must have 5 or 7 channels, got {}".format(materials.shape)
        assert (
            materials_jitter.shape[-1] == 5 or materials_jitter.shape[-1] == 7
        ), "materials_jitter must have 5 or 7 channels, got {}".format(
            materials_jitter.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        assert (
            sdf.shape == t_starts.shape
        ), "sdf must have shape of (N,)! Got {}".format(sdf.shape)
        assert (
            sdf_grad.shape[-1] == 3
        ), "sdf_grad must have 3 channels, got {}".format(sdf_grad.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        albedo, roughness, metallic = (
            materials[..., :3],
            materials[..., 3:4],
            materials[..., 4:],
        )
        albedo_jitter, roughness_jitter, metallic_jitter = (
            materials_jitter[..., :3],
            materials_jitter[..., 3:4],
            materials_jitter[..., 4:],
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
            "sigmas": sigmas,
            "normals": normals,
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "albedo_jitter": albedo_jitter,
            "roughness_jitter": roughness_jitter,
            "metallic_jitter": metallic_jitter,
            "sdf": sdf,
            "sdf_grad": sdf_grad,
        }
        if laplacian:
            extras.update({
                "sdf_laplace": sdf_laplace
            })
    # Rendering: accumulate rgbs, normals, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    normals = accumulate_along_rays(
        weights, values=normals, ray_indices=ray_indices, n_rays=n_rays
    )
    albedo = accumulate_along_rays(
        weights, values=albedo, ray_indices=ray_indices, n_rays=n_rays
    )
    roughness = accumulate_along_rays(
        weights, values=roughness, ray_indices=ray_indices, n_rays=n_rays
    )
    metallic = accumulate_along_rays(
        weights, values=metallic, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    # depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)
        normals = normals + render_bkgd * (1 - opacities) * torch.tensor(
            [0.0, 0.0, 1.0], device=normals.device
        )  # Background normal

    return colors, normals, albedo, roughness, metallic, opacities, depths, extras

def rendering_with_normals_mats_sdf_dir(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    material_dim = 5,
    laplacian=False
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).
        material_dim: dimension of material. Must be either 5 or 7.  5 = 3-albedo + 1-roughness +
            1-metallic. 7 = 3-albedo + 1-roughness + 3-metallic - in this case metallic means
            specular albedo.

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        raise NotImplementedError("rgb_sigma_fn is not implemented yet.")
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            if laplacian:
                (
                    rgbs,
                    normals,
                    materials,
                    materials_jitter,
                    alphas,
                    sigmas,
                    sdf,
                    sdf_grad,
                    t_dirs,
                    sdf_laplace
                ) = rgb_alpha_fn(t_starts, t_ends, ray_indices)
            else:
                (
                    rgbs,
                    normals,
                    materials,
                    materials_jitter,
                    alphas,
                    sigmas,
                    sdf,
                    sdf_grad,
                    t_dirs
                ) = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            normals = torch.empty((0, 3), device=t_starts.device)
            materials = torch.empty((0, material_dim), device=t_starts.device)
            materials_jitter = torch.empty((0, material_dim), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
            sdf = torch.empty((0,), device=t_starts.device)
            sdf_grad = torch.empty((0, 3), device=t_starts.device)
            sdf_laplace = torch.empty((0,), device=t_starts.device)
            t_dirs = torch.empty((0, 3), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert normals.shape[-1] == 3, "normals must have 3 channels, got {}".format(
            normals.shape
        )
        assert (
            materials.shape[-1] == 5 or materials.shape[-1] == 7
        ), "materials must have 5 or 7 channels, got {}".format(materials.shape)
        assert (
            materials_jitter.shape[-1] == 5 or materials_jitter.shape[-1] == 7
        ), "materials_jitter must have 5 or 7 channels, got {}".format(
            materials_jitter.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        assert (
            sdf.shape == t_starts.shape
        ), "sdf must have shape of (N,)! Got {}".format(sdf.shape)
        assert (
            sdf_grad.shape[-1] == 3
        ), "sdf_grad must have 3 channels, got {}".format(sdf_grad.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        albedo, roughness, metallic = (
            materials[..., :3],
            materials[..., 3:4],
            materials[..., 4:],
        )
        albedo_jitter, roughness_jitter, metallic_jitter = (
            materials_jitter[..., :3],
            materials_jitter[..., 3:4],
            materials_jitter[..., 4:],
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
            "sigmas": sigmas,
            "normals": normals,
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "albedo_jitter": albedo_jitter,
            "roughness_jitter": roughness_jitter,
            "metallic_jitter": metallic_jitter,
            "sdf": sdf,
            "sdf_grad": sdf_grad,
            "t_dirs": t_dirs
        }
        if laplacian:
            extras.update({
                "sdf_laplace": sdf_laplace
            })
    # Rendering: accumulate rgbs, normals, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    normals = accumulate_along_rays(
        weights, values=normals, ray_indices=ray_indices, n_rays=n_rays
    )
    albedo = accumulate_along_rays(
        weights, values=albedo, ray_indices=ray_indices, n_rays=n_rays
    )
    roughness = accumulate_along_rays(
        weights, values=roughness, ray_indices=ray_indices, n_rays=n_rays
    )
    metallic = accumulate_along_rays(
        weights, values=metallic, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    # depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)
        normals = normals + render_bkgd * (1 - opacities) * torch.tensor(
            [0.0, 0.0, 1.0], device=normals.device
        )  # Background normal

    return colors, normals, albedo, roughness, metallic, opacities, depths, extras