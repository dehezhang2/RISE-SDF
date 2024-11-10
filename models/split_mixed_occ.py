import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step

from nerfacc import (
    OccGridEstimator,
    accumulate_along_rays,
)
from lib.pbr import rgb_to_srgb
from models.volrend import rendering, rendering_with_normals_sdf, secondary_rendering
from lib.nerfacc import ContractionType


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter(
            "variance", nn.Parameter(torch.tensor(self.config.init_val))
        )
        self.modulate = self.config.get("modulate", False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s

    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s

    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min(
                    (global_step / self.reach_max_steps)
                    * (self.max_inv_s - self.prev_inv_s)
                    + self.prev_inv_s,
                    self.max_inv_s,
                )


@models.register("split-mixed-occ")
class SplitMixedOCCModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.emitter = models.make(self.config.light.name, self.config.light)

        self.geometry.contraction_type = ContractionType.AABB
        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer(
            "scene_aabb",
            torch.as_tensor(
                [
                    -self.config.radius,
                    -self.config.radius,
                    -self.config.radius,
                    self.config.radius,
                    self.config.radius,
                    self.config.radius,
                ],
                dtype=torch.float32,
            ),
        )
        if self.config.grid_prune:
            self.occupancy_grid = OccGridEstimator(
                roi_aabb=self.scene_aabb,
                resolution=128,
            )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = (
            1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        )
        
        self.num_samples_per_secondary_ray = self.config.get("num_samples_per_secondary_ray", 96)
        self.secondary_near_plane = self.config.get("secondary_near_plane", 0.05)
        self.secondary_far_plane = self.config.get("secondary_far_plane", 1.5)
        self.secondary_shader_chunk = self.config.get("secondary_shader_chunk", 160000)

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get("cos_anneal_end", 0)
        self.cos_anneal_ratio = (
            1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)
        )

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[..., None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[..., None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha

        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[..., None] * self.render_step_size_bg

        if self.training and self.config.grid_prune:
            self.occupancy_grid.update_every_n_steps(
                step=global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get("grid_prune_occ_thre", 0.01),
            )
        
        if global_step >= self.config.split_sum_kick_in_step:
            self.stage = 1
        else:
            self.stage = 0
        


    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def compute_relative_smoothness_loss(self, values, values_jittor):

        base = torch.maximum(values, values_jittor).clip(min=1e-6)
        difference = torch.sum(((values - values_jittor) / base)**2, dim=-1, keepdim=True)  # [..., 1]

        return difference

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(
            1e-6, 1e6
        )  # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
            + F.relu(-true_cos) * self.cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[..., None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[..., None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def compute_indirect_radiance(self, rays_o, rays_d):
        n_rays = rays_o.shape[0]

        def alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)
            sdf, sdf_grad = self.geometry(
                positions, with_grad=True, with_feature=False
            )
            normal = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            dists = (t_ends - t_starts)[..., None]
            alphas = self.get_alpha(sdf, normal, t_dirs, dists)
            return alphas

        with torch.no_grad():
            secondary_render_step_size = (
                self.secondary_far_plane - self.secondary_near_plane
            ) / (self.num_samples_per_secondary_ray - 1)
            ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
                rays_o,
                rays_d,
                alpha_fn=alpha_fn,
                near_plane=self.secondary_near_plane,
                far_plane=self.secondary_far_plane,
                render_step_size=secondary_render_step_size,
                stratified=False,
            )
            (
                acc_map,
                depth_map,
                _
            ) = secondary_rendering(
                t_starts,
                t_ends,
                ray_indices=ray_indices,
                n_rays=n_rays,
                alpha_fn=alpha_fn,
                chunk_size=self.secondary_shader_chunk,
            )

        return 1.0 - acc_map, depth_map

    def forward_(self, rays, relighting=False):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        def alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)
            sdf, sdf_grad = self.geometry(
                positions, with_grad=True, with_feature=False
            )
            normal = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            dists = (t_ends - t_starts)[..., None]
            alphas = self.get_alpha(sdf, normal, t_dirs, dists)
            return alphas

        def rgb_normal_alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=t_origins.device
                ), torch.zeros((0,), device=t_origins.device)
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            if self.config.geometry.grad_type == 'finite_difference' and self.training:
                sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
            else:
                sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            
            normal = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            dists = (t_ends - t_starts)[..., None]
            alphas = self.get_alpha(sdf, normal, t_dirs, dists)
            colors = self.texture(feature, t_dirs, normal, positions, self.emitter, self.stage)
            
            if self.config.geometry.grad_type == 'finite_difference' and self.training:
                return colors, normal, alphas, sdf, sdf_grad, sdf_laplace
            return colors, normal, alphas, sdf, sdf_grad

        ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
            rays_o,
            rays_d,
            alpha_fn=alpha_fn,
            render_step_size=self.render_step_size,
            stratified=self.randomized,
            cone_angle=0.0,
            alpha_thre=0.0,
        )

        (
            rgb_map,
            normal_map,
            acc_map,
            depth_map,
            extras
        ) = rendering_with_normals_sdf(
            t_starts,
            t_ends,
            ray_indices=ray_indices,
            n_rays=n_rays,
            rgb_alpha_fn=rgb_normal_alpha_fn,
            render_bkgd=None,
            has_laplace=(self.config.geometry.grad_type == 'finite_difference' and self.training), 
            color_dim = 7 if self.stage == 0 else 24
        )
        
        valid_indices = torch.nonzero(acc_map > 0.5)[..., 0]

        diff_rgb_map = rgb_map[..., :3]
        spec_rgb_map = rgb_map[..., 3:6]
        blend_map = rgb_map[..., 6:7]

        if self.stage != 0:
            diff_rgb_pbr_map = rgb_map[..., 7:10]
            spec_rgb_pbr_map = rgb_map[..., 10:13]
            spec_ref_map = rgb_map[..., 13:16]
            spec_light_map = rgb_map[..., 16:19]
            albedo_map = rgb_map[..., 19:22]
            metallic_map = rgb_map[..., 22:23]
            roughness_map = rgb_map[..., 23:]

        if valid_indices.numel() > 0 and self.config.indirect_pred:
            secondary_rays_o = rays_o[valid_indices] + depth_map[valid_indices] * rays_d[valid_indices]
            wo = -rays_d[valid_indices]
            secondary_rays_d = 2 * torch.sum(wo * normal_map[valid_indices], dim=-1, keepdim=True) * normal_map[valid_indices] - wo
            tr, secondary_depth= self.compute_indirect_radiance(secondary_rays_o, secondary_rays_d)
            tr = tr.clamp(0, 1)
            tr = tr.detach()
            secondary_depth = secondary_depth.detach()

            _, secondary_feature = self.geometry(secondary_rays_o, with_grad=False, with_feature=True)

            secondary_rgb = self.texture.secondary_shading(secondary_feature, secondary_rays_d, normal_map[valid_indices])
            spec_rgb_map[valid_indices] = tr * spec_rgb_map[valid_indices] +  (1 - tr) * secondary_rgb
            
            if self.stage != 0:
                if not relighting:
                    spec_rgb_pbr_map[valid_indices] = tr *  spec_rgb_pbr_map[valid_indices] + (1 - tr) * secondary_rgb
                else:
                    roughness_mask = (roughness_map[valid_indices] <= self.config.relighting_threshold)[..., 0]
                    third_rays_o = secondary_rays_o[roughness_mask] + secondary_depth[roughness_mask] * secondary_rays_d[roughness_mask]
                    _, third_grad, third_feature = self.geometry(third_rays_o, with_grad=True, with_feature=True)
                    third_normal =  F.normalize(third_grad, p=2, dim=-1, eps=1e-6)
                    secondary_rgb = self.texture.secondary_shading_pbr(third_feature, secondary_rays_d[roughness_mask], third_normal, third_rays_o, self.emitter)
                    spec_light_map_valid = spec_light_map[valid_indices]
                    spec_light_map_valid[roughness_mask] = tr[roughness_mask] * spec_light_map_valid[roughness_mask] + (1-tr[roughness_mask]) * secondary_rgb
                    spec_light_map[valid_indices] = spec_light_map_valid
                    spec_rgb_pbr_map = spec_ref_map * spec_light_map

        rgb_map = diff_rgb_map + spec_rgb_map
        
        if self.stage != 0:
            rgb_pbr_map = diff_rgb_pbr_map + spec_rgb_pbr_map


        out = {
            "comp_rgb": rgb_map,
            "comp_diffuse_rgb": diff_rgb_map,
            "comp_spec_rgb": spec_rgb_map,
            "comp_blend": blend_map,
            "comp_normal": normal_map,
            "opacity": acc_map,
            "depth": depth_map,
            "rays_valid": acc_map > 0,
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rays.device
            ),
        }

        if self.stage != 0:
            out.update({
                "comp_rgb_phys": rgb_pbr_map,
                "comp_diffuse_rgb_phys": diff_rgb_pbr_map,
                "comp_spec_rgb_phys": spec_rgb_pbr_map,
                "comp_albedo": albedo_map,
                "comp_metallic": metallic_map,
                "comp_roughness": roughness_map,
            })
        
        if self.training:
            weights = extras["weights"]
            sdf = extras["sdf"]
            sdf_grad = extras["sdf_grad"]
            out.update(
                {
                    "sdf_samples": sdf,
                    "sdf_grad_samples": sdf_grad,
                    "weights": weights.view(-1),
                    # TODO: following variables are useful for unbounded scenes
                    # "points": midpoints.view(-1),
                    # "intervals": dists.view(-1),
                    "ray_indices": ray_indices.view(-1),
                }
            )
            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': extras["sdf_laplace"]
                })
            # Normal orientation loss 
            if ray_indices.numel() > 0:
                normals = extras["normals"]
                normals_orientation_loss = torch.sum(
                    rays_d[ray_indices] * normals, dim=-1, keepdim=True
                ).clamp(min=0)
                normals_orientation_loss_map = accumulate_along_rays(
                    weights,
                    values=normals_orientation_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
            else:
                normals_orientation_loss_map = torch.zeros_like(rgb_map[..., :1])
            out.update(
                {
                    "normals_orientation_loss_map": normals_orientation_loss_map,
                }
            )


        if self.config.learned_background:
            raise NotImplementedError("Learned background not implemented.")
        else:
            out_bg = {
                "comp_rgb": self.background_color[None, :].expand(*rgb_map.shape),
                "num_samples": torch.zeros_like(out["num_samples"]),
                "rays_valid": torch.zeros_like(out["rays_valid"]), 
            }
            if self.stage != 0:
                out_bg.update({
                    "comp_rgb_phys": self.background_color[None, :].expand(*rgb_pbr_map.shape)
                })
        out_full = {
            "comp_rgb": rgb_to_srgb(
                out["comp_rgb"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])
            ).clamp(0, 1),
            "num_samples": out["num_samples"] + out_bg["num_samples"],
            "rays_valid": out["rays_valid"] | out_bg["rays_valid"],
        }

        if self.stage != 0:
            out_full.update({
                "comp_rgb_phys": rgb_to_srgb(
                    out["comp_rgb_phys"] + out_bg["comp_rgb_phys"] * (1.0 - out["opacity"])
                ).clamp(0, 1),

                "comp_spec_rgb": rgb_to_srgb(
                    out["comp_spec_rgb"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])
                ).clamp(0, 1),

                "comp_spec_rgb_phys": rgb_to_srgb(
                    out["comp_spec_rgb_phys"] + out_bg["comp_rgb_phys"] * (1.0 - out["opacity"])
                ).clamp(0, 1),
            })

        return {
            **out,
            **{k + "_bg": v for k, v in out_bg.items()},
            **{k + "_full": v for k, v in out_full.items()},
        }

    def forward(self, rays, relighting = False):
        if self.training:
            out = self.forward_(rays, relighting=relighting)
        else:
            out = chunk_batch(
                self.forward_,
                self.config.ray_chunk,
                True,
                rays,
                relighting
            )
        return {**out, "inv_s": self.variance.inv_s}

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(
                self.geometry,
                export_config.chunk_size,
                False,
                mesh["v_pos"].to(self.rank),
                with_grad=True,
                with_feature=True,
            )
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            with torch.no_grad():
                colors = self.texture(feature, -normal, normal, mesh["v_pos"].to(self.rank), self.emitter, self.stage)
            colors = colors.detach()
            albedo = colors[..., 19:22]
            metallic = colors[..., 22:23]
            roughness = colors[..., 23:]
            # breakpoint()
            # mesh["v_rgb"] = colors[:, :3].cpu()
        return mesh, albedo, metallic, roughness
