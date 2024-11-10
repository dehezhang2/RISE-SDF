import torch
import torch.nn as nn

import models
from models.geometry import contract_to_unisphere
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step

import nvdiffrast.torch as dr
import numpy as np
import torch.nn.functional as F
# from lib.pbr.utils.warp_utils import coordinate_system, sample_GGX_VNDF, to_world, sample_uniform_cylinder, to_local

@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding = encoding
        self.network = network

    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network

    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}

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
    else:
        raise NotImplementedError


def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps) ** (12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps) ** (12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError

@models.register('volume-split-sum-mip-occ')
class VolumeSplitSumMip(nn.Module):

    def __init__(self, config):
        super(VolumeSplitSumMip, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_pos_dims = self.config.get('n_pos_dims', 3)
        self.n_output_dims = 3
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        xyz_encoding = get_encoding(self.n_pos_dims, self.config.xyz_encoding_config)
        self.dir_encoding = dir_encoding
        self.xyz_encoding = xyz_encoding
        
        secondary_network = get_mlp(
            self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims, 
            3, 
            self.config.secondary_mlp_network_config
        )

        albedo_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims, 
            3, 
            self.config.albedo_mlp_network_config
        )

        roughness_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims, 
            1, 
            self.config.roughness_mlp_network_config
        )

        metallic_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims, 
            1, 
            self.config.metallic_mlp_network_config
        )

        self.secondary_network = secondary_network
        self.albedo_network = albedo_network
        self.roughness_network = roughness_network
        self.metallic_network = metallic_network

        # precomputed microfacet integration
        FG_LUT = torch.from_numpy(np.fromfile('load/bsdf/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)
        self.FG_LUT.requires_grad_(False)
        
        # self.sample = torch.rand(self.config.sample_size, 2).cuda()

    def forward(self, features, dirs, normals, positions, emitter, *args):
        if dirs.shape[0] == 0:           
            return torch.zeros((0, 3)).cuda()
        wi = -dirs
        wo = torch.sum(wi * normals, -1, keepdim=True) * normals * 2 - wi
        NoV = torch.sum(normals * wi, -1, keepdim=True)
        xyz_embd = self.xyz_encoding(positions.view(-1, self.n_pos_dims))
        
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        
        albedo = self.albedo_network(network_inp).view(*features.shape[:-1], 3).float()

        roughness = self.roughness_network(network_inp).view(*features.shape[:-1], 1).float()
        
        metallic = self.metallic_network(network_inp).view(*features.shape[:-1], 1).float()
        
        if 'color_activation' in self.config:
            albedo = get_activation(self.config.color_activation)(albedo)
            metallic = get_activation(self.config.color_activation)(metallic)
            roughness = get_activation(self.config.color_activation)(roughness)

        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = emitter.eval_mip(normals)
        diff_rgb_pbr = diffuse_albedo * diffuse_light

            
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light = emitter.eval_mip(wo, specular=True, roughness=roughness)
            
        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = dirs.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, fg_uv.shape[-1]).contiguous(), filter_mode='linear',
                            boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        spec_rgb_pbr = specular_ref * specular_light
        return torch.cat([diff_rgb_pbr, spec_rgb_pbr, specular_ref, specular_light, albedo, metallic, roughness], dim=-1)

    def secondary_shading(self, features, rays_d, *args):
        rays_d = (rays_d + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(rays_d.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.secondary_network(network_inp).view(*network_inp.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def secondary_shading_pbr(self, features, dirs, normals, positions, emitter):
        if dirs.shape[0] == 0:           
            return torch.zeros((0, 3)).cuda()
        
        wi = -dirs
        wo = torch.sum(wi * normals, -1, keepdim=True) * normals * 2 - wi
        NoV = torch.sum(normals * wi, -1, keepdim=True)
        
        xyz_embd = self.xyz_encoding(positions.view(-1, self.n_pos_dims))
        
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        albedo = self.albedo_network(network_inp).view(*features.shape[:-1], 3).float()
        roughness = self.roughness_network(network_inp).view(*features.shape[:-1], 1).float()
        metallic = self.metallic_network(network_inp).view(*features.shape[:-1], 1).float()
        
        if 'color_activation' in self.config:
            albedo = get_activation(self.config.color_activation)(albedo)
            metallic = get_activation(self.config.color_activation)(metallic)
            roughness = get_activation(self.config.color_activation)(roughness)

        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = emitter.eval_mip(normals)
        diff_rgb_pbr = diffuse_albedo * diffuse_light

            
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light = emitter.eval_mip(dirs, specular=True, roughness=roughness)
        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = dirs.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, fg_uv.shape[-1]).contiguous(), filter_mode='linear',
                            boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        spec_rgb_pbr = specular_ref * specular_light
        return diff_rgb_pbr + spec_rgb_pbr
    
    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        return {}

@models.register('volume-mixed-mip-split-occ')
class VolumeMixedMipSplitOcc(nn.Module):

    def __init__(self, config):
        super(VolumeMixedMipSplitOcc, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_pos_dims = self.config.get('n_pos_dims', 3)
        self.n_output_dims = 3
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        xyz_encoding = get_encoding(self.n_pos_dims, self.config.xyz_encoding_config)
        self.dir_encoding = dir_encoding
        self.xyz_encoding = xyz_encoding
        
        secondary_network = get_mlp(
            self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims, 
            3, 
            self.config.secondary_mlp_network_config
        )

        albedo_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims, 
            6, 
            self.config.albedo_mlp_network_config
        )

        roughness_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims, 
            1, 
            self.config.roughness_mlp_network_config
        )
        
        env_network = get_mlp(
            self.config.input_feature_dim + dir_encoding.n_output_dims, 
            3, 
            self.config.spec_mlp_network_config
        )

        metallic_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims, 
            2, 
            self.config.metallic_mlp_network_config
        )

        self.secondary_network = secondary_network
        self.albedo_network = albedo_network
        self.roughness_network = roughness_network
        self.metallic_network = metallic_network
        self.env_network = env_network

        # precomputed microfacet integration
        FG_LUT = torch.from_numpy(np.fromfile('load/bsdf/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)
        self.FG_LUT.requires_grad_(False)
        
        # self.sample = torch.rand(self.config.sample_size, 2).cuda()
        
    
    def forward(self, features, dirs, normals, positions, emitter, stage = 0, *args):
        if dirs.shape[0] == 0:           
            return torch.zeros((0, 3)).cuda()
        wi = -dirs
        wo = torch.sum(wi * normals, -1, keepdim=True) * normals * 2 - wi
        NoV = torch.sum(normals * wi, -1, keepdim=True)
        xyz_embd = self.xyz_encoding(positions.view(-1, self.n_pos_dims))
        
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        
        albedo = self.albedo_network(network_inp).view(*features.shape[:-1], 6).float()
        diff_rgb = albedo[..., :3]
        albedo = albedo[..., 3:]

        roughness = self.roughness_network(network_inp).view(*features.shape[:-1], 1).float()
        
        metallic = self.metallic_network(network_inp).view(*features.shape[:-1], 2).float()
        blend = metallic[..., :1]
        metallic = metallic[..., 1:]

        wo_enc = self.dir_encoding(((wo + 1.) / 2.).view(-1, self.n_dir_dims))
        env_network_inp = torch.cat([features, wo_enc], dim=-1)
        spec_rgb = self.env_network(env_network_inp).view(*features.shape[:-1], 3).float()
        
        if 'color_activation' in self.config:
            albedo = get_activation(self.config.color_activation)(albedo)
            diff_rgb = get_activation(self.config.color_activation)(diff_rgb)
            blend = get_activation(self.config.color_activation)(blend)
            metallic = get_activation(self.config.color_activation)(metallic)
            roughness = get_activation(self.config.color_activation)(roughness)
            spec_rgb = get_activation(self.config.color_activation)(spec_rgb)
        spec_rgb = blend * spec_rgb
        diff_rgb = (1 - blend) * diff_rgb

        if stage == 0:
            return torch.cat([diff_rgb, spec_rgb, blend], dim=-1)

        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = emitter.eval_mip(normals)
       
        diff_rgb_pbr = diffuse_albedo * diffuse_light

            
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light = emitter.eval_mip(wo, specular=True, roughness=roughness)
            
        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = dirs.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, fg_uv.shape[-1]).contiguous(), filter_mode='linear',
                            boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        spec_rgb_pbr = specular_ref * specular_light
        # return torch.cat([diff_rgb, spec_rgb, blend, diff_rgb_pbr, spec_rgb_pbr, albedo, metallic, roughness], dim=-1)
        return torch.cat([diff_rgb, spec_rgb, blend, diff_rgb_pbr, spec_rgb_pbr, specular_ref, specular_light, albedo, metallic, roughness], dim=-1)

    def secondary_shading(self, features, rays_d, *args):
        rays_d = (rays_d + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(rays_d.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.secondary_network(network_inp).view(*network_inp.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def secondary_shading_radiance(self, features, dirs, normals, positions):
        if dirs.shape[0] == 0:           
            return torch.zeros((0, 3)).cuda()
        wi = -dirs
        wo = torch.sum(wi * normals, -1, keepdim=True) * normals * 2 - wi
        NoV = torch.sum(normals * wi, -1, keepdim=True)
        xyz_embd = self.xyz_encoding(positions.view(-1, self.n_pos_dims))
        
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        
        albedo = self.albedo_network(network_inp).view(*features.shape[:-1], 6).float()
        diff_rgb = albedo[..., :3]
        
        metallic = self.metallic_network(network_inp).view(*features.shape[:-1], 2).float()
        blend = metallic[..., :1]

        wo_enc = self.dir_encoding(((wo + 1.) / 2.).view(-1, self.n_dir_dims))
        env_network_inp = torch.cat([features, wo_enc], dim=-1)
        spec_rgb = self.env_network(env_network_inp).view(*features.shape[:-1], 3).float()
        
        if 'color_activation' in self.config:
            diff_rgb = get_activation(self.config.color_activation)(diff_rgb)
            blend = get_activation(self.config.color_activation)(blend)
            spec_rgb = get_activation(self.config.color_activation)(spec_rgb)

        spec_rgb = blend * spec_rgb
        diff_rgb = (1 - blend) * diff_rgb

        return diff_rgb + spec_rgb

    def secondary_shading_pbr(self, features, dirs, normals, positions, emitter):
        if dirs.shape[0] == 0:           
            return torch.zeros((0, 3)).cuda()
        
        wi = -dirs
        wo = torch.sum(wi * normals, -1, keepdim=True) * normals * 2 - wi
        NoV = torch.sum(normals * wi, -1, keepdim=True)
        
        xyz_embd = self.xyz_encoding(positions.view(-1, self.n_pos_dims))
        
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        
        albedo = self.albedo_network(network_inp).view(*features.shape[:-1], 6).float()
        albedo = albedo[..., 3:]

        roughness = self.roughness_network(network_inp).view(*features.shape[:-1], 1).float()
        
        metallic = self.metallic_network(network_inp).view(*features.shape[:-1], 2).float()
        metallic = metallic[..., 1:]
        
        if 'color_activation' in self.config:
            albedo = get_activation(self.config.color_activation)(albedo)
            metallic = get_activation(self.config.color_activation)(metallic)
            roughness = get_activation(self.config.color_activation)(roughness)

        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = emitter.eval_mip(normals)
       
        diff_rgb_pbr = diffuse_albedo * diffuse_light

            
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light = emitter.eval_mip(dirs, specular=True, roughness=roughness)
            
        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = dirs.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, fg_uv.shape[-1]).contiguous(), filter_mode='linear',
                            boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        spec_rgb_pbr = specular_ref * specular_light

        return diff_rgb_pbr + spec_rgb_pbr
    
    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        return {}

@models.register('volume-pbr')
class VolumePBR(nn.Module):

    def __init__(self, config):
        super(VolumePBR, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.scatterer = models.make(self.config.scatterer.name, self.config.scatterer)

    def forward_mats(self, positions, dirs, normals, 
                     albedo, roughness, metallic, 
                     emitter, compute_indirect):
        wi = -dirs
        
        attenuation = torch.zeros_like(roughness)
        with torch.no_grad():
            # Scatterer sampling
            secondary_rays_d = self.scatterer.sample(
                n=normals,
                wi=wi,
                alpha_x=roughness.squeeze(-1),
                alpha_y=roughness.squeeze(-1),
                albedo=albedo,
                metallic=metallic,
                attenuation=attenuation,
            )
            secondary_rays_o = positions.reshape(-1, 3)
            secondary_tr, secondary_rgb_map = compute_indirect(
                secondary_rays_o,
                secondary_rays_d,
            )
        pdf = self.scatterer.pdf(
            n=normals,
            wi=wi,
            wo=secondary_rays_d,
            alpha_x=roughness.squeeze(-1),
            alpha_y=roughness.squeeze(-1),
            albedo=albedo,
            metallic=metallic,
            attenuation=attenuation,
        )
        pdf = torch.where(pdf > 0, pdf, torch.ones_like(pdf))
        diff, spec = self.scatterer.eval(
            wi=wi,
            n=normals,
            wo=secondary_rays_d,
            alpha_x=roughness.squeeze(-1),
            alpha_y=roughness.squeeze(-1),
            albedo=albedo,
            metallic=metallic,
            attenuation=attenuation,
        )
        em_li = emitter.eval(secondary_rays_d)
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        Lo_diff = Li * diff / pdf
        Lo_spec = Li * spec / pdf
        
        if metallic.size(-1) == 1:
            kd = (1.0 - metallic) * albedo
            ks = torch.ones_like(kd)
        else:
            kd = albedo
            ks = metallic

        Lo_diff = kd * Lo_diff
        Lo_spec = ks * Lo_spec
        Lo =  Lo_diff + Lo_spec

        return Lo, Lo_diff, Lo_spec
        
    def forward(self, 
                positions, dirs, normals, 
                albedo, roughness, metallic, 
                compute_indirect, emitter,
                *args):
        results = {}
        if dirs.shape[0] == 0:
            results.update({
                'rgb_phys':   torch.zeros((0, 3)).cuda(),
                'specular_color': torch.zeros((0, 3)).cuda(),
                'diffuse_color': torch.zeros((0, 3)).cuda(),
            })
            return results
    
        Lo, Lo_diff, Lo_spec = self.forward_mats(positions, dirs, normals, 
                                                 albedo, roughness, metallic, 
                                                 emitter, compute_indirect)
        results.update({
            'rgb_phys':  Lo,
            'specular_color': Lo_spec,
            'diffuse_color': Lo_diff
        })
        
        return results

    def regularizations(self, out):
        return {}
    