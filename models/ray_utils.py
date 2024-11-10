import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(
    W, H, fx, fy, cx, cy, use_pixel_centers=True, openGL_camera=True
):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack(
        [
            (i - cx) / fx,
            -(j - cy) / fy,
            -torch.ones_like(i) if openGL_camera else torch.ones_like(i),
        ],
        -1,
    )  # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False, opencv_format=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if not opencv_format:
        if directions.ndim == 2:  # (N_rays, 3)
            assert c2w.ndim == 3  # (N_rays, 3, 4) / (1, 3, 4)
            rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
            rays_o = c2w[:, :, 3].expand(rays_d.shape)
        elif directions.ndim == 3:  # (H, W, 3)
            if c2w.ndim == 2:  # (3, 4)
                rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                    -1
                )  # (H, W, 3)
                rays_o = c2w[None, None, :, 3].expand(rays_d.shape)
            elif c2w.ndim == 3:  # (B, 3, 4)
                rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                    -1
                )  # (B, H, W, 3)
                rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)

        if not keepdim:
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    else:
        if c2w.ndim == 2:
            c2w = c2w.unsqueeze(0)
        rays_d = c2w[..., :3].permute(0, 2, 1) @ directions.unsqueeze(-1)
        rays_d = rays_d[..., 0]
        rays_o = c2w[..., :3].permute(0, 2, 1) @ -c2w[..., 3:]  # trn,3,1
        rays_o = rays_o[..., 0]  # rn,3
        rays_o = rays_o.expand(rays_d.shape)

        if not keepdim:
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
