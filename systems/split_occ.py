import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
import numpy as np
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy, MAE
from lib.pbr import rgb_to_srgb
import os

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    
def HWC_to_BCHW(HWC):
    BCHW = HWC.unsqueeze(0).permute(0, 3, 1, 2)
    return BCHW

@systems.register('split-occ-system')

class SplitOccSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'mae' : MAE
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        

    def forward(self, batch, relighting=False):
        return self.model(batch['rays'], relighting=relighting)
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)

        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            
            if stage in ['test']:
                if self.config.dataset.has_albedo:
                    albedo = self.dataset.all_albedo[index.to(self.dataset.all_albedo.device)].view(-1, self.dataset.all_albedo.shape[-1]).to(self.rank)
                if self.config.dataset.has_roughness:
                    roughness = self.dataset.all_roughness[index.to(self.dataset.all_roughness.device)].view(-1, self.dataset.all_roughness.shape[-1]).to(self.rank)
                normal = self.dataset.all_normals[index.to(self.dataset.all_normals.device)].view(-1, self.dataset.all_normals.shape[-1]).to(self.rank)
                relight_rgb = {}
                for light in self.config.dataset.relight_list:
                    relight_rgb[light] = self.dataset.relight_images[light][index.to(self.dataset.all_normals.device)].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
                

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'black':
                self.model.background_color = torch.zeros((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)

        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[..., None] + rgb_to_srgb(
                self.model.background_color * (1 - fg_mask[..., None])
            )
            if stage in ['test']:
                for light in self.config.dataset.relight_list:
                    relight_rgb[light] = relight_rgb[light] * fg_mask[..., None] + rgb_to_srgb(
                    self.model.background_color * (1 - fg_mask[..., None])
                )

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })

        if stage in ['test']:
            batch.update({
                'relight_rgb': relight_rgb,
                'normal' : normal
            })
            if self.config.dataset.has_albedo:
                 batch.update({
                    'albedo' : albedo,
                 })

            if self.config.dataset.has_roughness:
                 batch.update({
                    'roughness' : roughness,
                 })


    
    def training_step(self, batch, batch_idx):
        if hasattr(self.model, 'emitter'):
            self.model.emitter.build_mips()

        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        if self.model.stage != 0:
            loss_rgb_phys_mse = F.mse_loss(out['comp_rgb_phys_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb_phys_mse', loss_rgb_phys_mse)
            loss += loss_rgb_phys_mse * self.C(self.config.system.loss.lambda_rgb_phys_mse)

            loss_rgb_phys_l1 = F.l1_loss(out['comp_rgb_phys_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb_phys', loss_rgb_phys_l1)
            loss += loss_rgb_phys_l1 * self.C(self.config.system.loss.lambda_rgb_phys_l1)


        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)
        
        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)
        
        if self.C(self.config.system.loss.lambda_emitter_distillation) > 0 and self.model.stage != 0:
            loss_emitter_distillation = F.mse_loss(out['comp_spec_rgb_full'][out['rays_valid_full'][...,0]], out['comp_spec_rgb_phys_full'][out['rays_valid_full'][...,0]])
            self.log('train/loss_emitter_distillation', loss_emitter_distillation)
            loss += loss_emitter_distillation * self.C(self.config.system.loss.lambda_emitter_distillation)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        if hasattr(self.model, 'emitter'):
            with torch.no_grad():
                self.model.emitter.build_mips()
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        phys_psnr = self.criterions['psnr'](out['comp_rgb_phys_full'].to(batch['rgb']), batch['rgb']) if self.model.stage != 0 else 0
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
             {'type': 'grayscale', 'img': out['comp_blend'].view(H, W), 'kwargs': {'data_range': None, 'cmap': None}},
            {'type': 'rgb', 'img': out['comp_spec_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_diffuse_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])

        if self.model.stage != 0:
            self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}-pbr.png", [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb_phys_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
            ] + ([
                {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ] if self.config.model.learned_background else []) + [
                {'type': 'rgb', 'img': out['comp_albedo'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': out['comp_metallic'].view(H, W), 'kwargs': {'data_range': None, 'cmap': None}},
                {'type': 'grayscale', 'img': out['comp_roughness'].view(H, W), 'kwargs': {'data_range': None, 'cmap': None}},
                {'type': 'rgb', 'img': out['comp_spec_rgb_phys'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_diffuse_rgb_phys'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ])

        res = {
             'psnr': psnr,
            'index': batch['index']
        }
        if self.model.stage != 0:
            res.update({
                'phys_psnr': phys_psnr,
            })
        return res
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {
                        'psnr': step_out['psnr'],
                        'phys_psnr': step_out['phys_psnr'] if self.model.stage != 0 else 0
                    }
                    
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {
                            'psnr': step_out['psnr'][oi],
                            'phys_psnr': step_out['phys_psnr'][oi] if self.model.stage != 0 else 0
                        }
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)        
            if self.model.stage != 0:
                phys_psnr = torch.mean(torch.stack([o['phys_psnr'] for o in out_set.values()]))
                self.log('val/phys_psnr', phys_psnr, prog_bar=True, rank_zero_only=True) 
            
                

    def test_step(self, batch, batch_idx):
        if hasattr(self.model, 'emitter'):            
            with torch.no_grad():
                self.model.emitter.build_mips()
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        phys_psnr = self.criterions['psnr'](out['comp_rgb_phys_full'].to(batch['rgb']), batch['rgb']) if self.model.stage != 0 else 0
        phys_ssim = SSIM(out['comp_rgb_phys_full'].to(batch['rgb']).reshape(1, 3, 800, 800), batch['rgb'].reshape(1, 3, 800, 800)) if self.model.stage != 0 else 0
        phys_lpips = LPIPS(out['comp_rgb_phys_full'].to(batch['rgb']).reshape(1, 3, 800, 800), batch['rgb'].reshape(1, 3, 800, 800)) if self.model.stage != 0 else 0
        
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
             {'type': 'grayscale', 'img': out['comp_blend'].view(H, W), 'kwargs': {'data_range': None, 'cmap': None}},
            {'type': 'rgb', 'img': out['comp_spec_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_diffuse_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])

        if self.model.stage != 0:
            self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}-pbr.png", [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb_phys_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
            ] + ([
                {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ] if self.config.model.learned_background else []) + [
                {'type': 'rgb', 'img': out['comp_albedo'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': out['comp_metallic'].view(H, W), 'kwargs': {'data_range': None, 'cmap': None}},
                {'type': 'grayscale', 'img': out['comp_roughness'].view(H, W), 'kwargs': {'data_range': None, 'cmap': None}},
                {'type': 'rgb', 'img': out['comp_spec_rgb_phys'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_diffuse_rgb_phys'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ])
            
            if self.config.dataset.has_albedo:
                # Apply alignment of the albedo
                albedo_map = out['comp_albedo'].to(batch['albedo'])
                gt_mask_reshaped = batch['fg_mask'].bool()
                ratio_value, _ = (batch['albedo'][gt_mask_reshaped]/ albedo_map[gt_mask_reshaped].clamp(min=1e-6)).median(dim=0)
                albedo_map[gt_mask_reshaped] = (ratio_value * albedo_map[gt_mask_reshaped]).clamp(min=0.0, max=1.0)
                
                albedo_psnr = self.criterions['psnr'](albedo_map, batch['albedo'])
                albedo_ssim = SSIM(albedo_map.reshape(1, 3, 800, 800), batch['albedo'].reshape(1, 3, 800, 800))
                albedo_lpips = LPIPS(albedo_map.reshape(1, 3, 800, 800), batch['albedo'].reshape(1, 3, 800, 800))

                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}-albedo.png", [
                    {'type': 'rgb', 'img': batch['albedo'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                    {'type': 'rgb', 'img': albedo_map.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                ])
            if self.config.dataset.has_roughness:
                # Apply alighment of the roughness
                roughness_map = out['comp_roughness'].to(batch['roughness'])
                ratio_value, _ = (batch['roughness'][gt_mask_reshaped]/ roughness_map[gt_mask_reshaped].clamp(min=1e-6)).median(dim=0)
                roughness_map[gt_mask_reshaped] = (ratio_value * roughness_map[gt_mask_reshaped]).clamp(min=0.0, max=1.0)
                
                roughness_psnr = self.criterions['psnr'](roughness_map, batch['roughness'])
                # self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}-roughness.png", [
                #     {'type': 'grayscale', 'img': batch['roughness'].view(H, W).float(), 'kwargs': {'data_range': None, 'cmap': None}},
                #     {'type': 'grayscale', 'img': out['comp_roughness'].view(H, W).float(), 'kwargs': {'data_range': None, 'cmap': None}},
                # ])
                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}-roughness.exr", [
                    {'type': 'hdr', 'img': batch['roughness'].view(H, W, 1), 'kwargs': {'data_format': 'HWC'}},
                    {'type': 'hdr', 'img': out['comp_roughness'].view(H, W, 1), 'kwargs': {'data_format': 'HWC'}},
                ])

            relight_psnr = {}
            relight_ssim = {}
            relight_lpips = {}
            emitter = self.model.emitter
            for light in self.config.dataset.relight_list:
                self.config.model.light.envlight_config.hdr_filepath = os.path.join(
                    self.config.dataset.hdr_filepath, f"{light}.hdr"
                ) if self.config.dataset.name == 'tensoir' else os.path.join(
                    self.config.dataset.hdr_filepath, f"{light}.exr"
                )
                self.model.emitter = models.make(self.config.model.light.name, self.config.model.light)
                with torch.no_grad():
                    self.model.emitter.build_mips()
                out = self(batch, relighting = True)

                pred_light = out['comp_rgb_phys_full'].to(batch['relight_rgb'][light])
                gt_mask_reshaped = batch['fg_mask'].bool()
                ratio_value, _ = (batch['relight_rgb'][light][gt_mask_reshaped]/ pred_light[gt_mask_reshaped].clamp(min=1e-6)).median(dim=0)
                pred_light[gt_mask_reshaped] = (ratio_value * pred_light[gt_mask_reshaped]).clamp(min=0.0, max=1.0)

                relight_psnr[light] = self.criterions['psnr'](pred_light, batch['relight_rgb'][light]) if self.model.stage != 0 else 0
                relight_ssim[light] = SSIM(pred_light.reshape(1, 3, 800, 800), batch['relight_rgb'][light].reshape(1, 3, 800, 800))
                relight_lpips[light] = LPIPS(pred_light.reshape(1, 3, 800, 800), batch['relight_rgb'][light].reshape(1, 3, 800, 800))

                self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}-relight-" + light+ ".png", [
                    {'type': 'rgb', 'img': batch['relight_rgb'][light].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                    {'type': 'rgb', 'img': out['comp_rgb_phys_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
                ])
            self.model.emitter = emitter
        res = {
            'psnr': psnr,
            'index': batch['index']
        }
        if self.model.stage != 0:
            res.update({
                'phys_psnr': phys_psnr,
                'phys_ssim': phys_ssim,
                'phys_lpips': phys_lpips,
            })
            if len(self.config.dataset.relight_list) != 0:
                res.update({
                    'relight_psnr': relight_psnr,
                    'relight_ssim': relight_ssim,
                    'relight_lpips': relight_lpips,
                })
            if self.config.dataset.has_albedo:
                 res.update({
                    'albedo_psnr': albedo_psnr,
                    'albedo_ssim': albedo_ssim,
                    'albedo_lpips': albedo_lpips,
                 })

            if self.config.dataset.has_roughness:
                 res.update({
                    'roughness_psnr': roughness_psnr,
                 })
        return res
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {
                        'psnr': step_out['psnr'],
                        'phys_psnr': step_out['phys_psnr'] if self.model.stage != 0 else 0,
                        'phys_ssim': step_out['phys_ssim'] if self.model.stage != 0 else 0,
                        'phys_lpips': step_out['phys_lpips'] if self.model.stage != 0 else 0,
                        'albedo_psnr': step_out['albedo_psnr'] if self.model.stage != 0 and self.config.dataset.has_albedo else 0,
                        'albedo_ssim': step_out['albedo_ssim'] if self.model.stage != 0 and self.config.dataset.has_albedo else 0,
                        'albedo_lpips': step_out['albedo_lpips'] if self.model.stage != 0 and self.config.dataset.has_albedo else 0,
                        'roughness_psnr': step_out['roughness_psnr'] if self.model.stage != 0 and self.config.dataset.has_roughness else 0,
                        'relight_psnr': step_out['relight_psnr'] if self.model.stage != 0 and len(self.config.dataset.relight_list)!=0 else 0,
                        'relight_ssim': step_out['relight_ssim'] if self.model.stage != 0 and len(self.config.dataset.relight_list)!=0 else 0,
                        'relight_lpips': step_out['relight_lpips'] if self.model.stage != 0 and len(self.config.dataset.relight_list)!=0 else 0
                    }
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {
                            'psnr': step_out['psnr'][oi],
                            'phys_psnr': step_out['phys_psnr'][oi] if self.model.stage != 0 else 0,
                            'phys_ssim': step_out['phys_ssim'][oi] if self.model.stage != 0 else 0,
                            'phys_lpips': step_out['phys_lpips'][oi] if self.model.stage != 0 else 0,
                            'albedo_psnr': step_out['albedo_psnr'][oi] if self.model.stage != 0 and self.config.dataset.has_albedo else 0,
                            'albedo_ssim': step_out['albedo_ssim'][oi] if self.model.stage != 0 and self.config.dataset.has_albedo else 0,
                            'albedo_lpips': step_out['albedo_lpips'][oi] if self.model.stage != 0 and self.config.dataset.has_albedo else 0,
                            'roughness_psnr': step_out['roughness_psnr'][oi] if self.model.stage != 0 and self.config.dataset.has_roughness else 0,
                            'relight_psnr': step_out['relight_psnr'][oi] if self.model.stage != 0 and len(self.config.dataset.relight_list)!=0 else 0,
                            'relight_ssim': step_out['relight_ssim'][oi] if self.model.stage != 0 and len(self.config.dataset.relight_list)!=0 else 0,
                            'relight_lpips': step_out['relight_lpips'][oi] if self.model.stage != 0 and len(self.config.dataset.relight_list)!=0 else 0,
                        }
            

            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    
            
            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )

            if self.model.stage != 0:
                phys_psnr = torch.mean(torch.stack([o['phys_psnr'] for o in out_set.values()]))
                phys_ssim = torch.mean(torch.stack([o['phys_ssim'] for o in out_set.values()]))
                phys_lpips = torch.mean(torch.stack([o['phys_lpips'] for o in out_set.values()]))
                self.log('test/phys_psnr', phys_psnr, prog_bar=True, rank_zero_only=True)
                self.log('test/phys_ssim', phys_ssim, prog_bar=True, rank_zero_only=True)
                self.log('test/phys_lpips', phys_lpips, prog_bar=True, rank_zero_only=True)

                if self.config.dataset.has_albedo:
                    albedo_psnr = torch.mean(torch.stack([o['albedo_psnr'] for o in out_set.values()]))
                    albedo_ssim = torch.mean(torch.stack([o['albedo_ssim'] for o in out_set.values()]))
                    albedo_lpips = torch.mean(torch.stack([o['albedo_lpips'] for o in out_set.values()]))
                    self.log('test/albedo_psnr', albedo_psnr, prog_bar=True, rank_zero_only=True)
                    self.log('test/albedo_ssim', albedo_ssim, prog_bar=True, rank_zero_only=True)
                    self.log('test/albedo_lpips', albedo_lpips, prog_bar=True, rank_zero_only=True)

                if self.config.dataset.has_roughness:
                    roughness_psnr = torch.mean(torch.stack([o['roughness_psnr'] for o in out_set.values()]))
                    self.log('test/roughness_psnr', roughness_psnr, prog_bar=True, rank_zero_only=True)


                for light in self.config.dataset.relight_list:
                    relight_psnr = torch.mean(torch.stack([o['relight_psnr'][light] for o in out_set.values()]))
                    relight_ssim = torch.mean(torch.stack([o['relight_ssim'][light] for o in out_set.values()]))
                    relight_lpips = torch.mean(torch.stack([o['relight_lpips'][light] for o in out_set.values()]))

                    self.log('test/relight_psnr_' + light, relight_psnr, prog_bar=True, rank_zero_only=True)
                    self.log('test/relight_ssim_' + light, relight_ssim, prog_bar=True, rank_zero_only=True)
                    self.log('test/relight_lpips_' + light, relight_lpips, prog_bar=True, rank_zero_only=True)

                self.save_img_sequence(
                    f"it{self.global_step}-test-pbr",
                    f"it{self.global_step}-test",
                    '(\d+)\-pbr.png',
                    save_format='mp4',
                    fps=30
                )
               
            # self.export()
    
    def export(self):
        mesh, albedo, metallic, roughness = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
            **mesh
        )
        np.save(self.get_save_path('metallic.npy'), linear_to_srgb(metallic.cpu().numpy()))
        np.save(self.get_save_path('roughness.npy'), linear_to_srgb(roughness.cpu().numpy()))
        np.save(self.get_save_path('albedo.npy'), linear_to_srgb(albedo.cpu().numpy()))        
