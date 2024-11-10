import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.geometry import contract_to_unisphere
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step

def get_camera_plane_intersection(pts, dirs, poses):
    """
    Source: https://github.com/liuyuan-pal/NeRO/blob/c210fe80aa9e6a590946a1469f2515f1e168495e/network/field.py#L348 
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:,:,:3], poses[:,:,3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:,:,None] + t)[..., 0] # pn,3
    dirs_ = (R @ dirs[:,:,None])[..., 0]   # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits

@models.register('volume-layered-radiance')
class VolumeLayeredRadiance(nn.Module):
    """
    Occlusion-aware volume radiance
    """
    def __init__(self, config):
        super(VolumeLayeredRadiance, self).__init__()
        self.config = config
        self.blend_type = self.config.get('blend_type', 'surface')
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_pos_dims = self.config.get('n_pos_dims', 3)
        self.n_output_dims = self.config.get('n_output_dims', 7)
        self.pred_vis = self.config.get('pred_vis', False)
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        xyz_encoding = get_encoding(self.n_pos_dims, self.config.xyz_encoding_config)
        ref_primary_network = get_mlp(
            self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims, 
            3, 
            self.config.ref_primary_network_config
        )
        ref_secondary_network = get_mlp(
            self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims + xyz_encoding.n_output_dims,
            # self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims, 
            4 if self.config.pred_vis else 3, 
            self.config.ref_secondary_network_config
        )
        dif_network = get_mlp(
            self.config.input_feature_dim,
            3, 
            self.config.diffuse_network_config
        )
        blend_network = get_mlp(
            self.config.input_feature_dim,
            1, 
            self.config.blend_network_config
        )
        self.dir_encoding = dir_encoding
        self.xyz_encoding = xyz_encoding
        self.ref_primary_network = ref_primary_network 

        if self.config.secondary_shading:
            self.ref_secondary_network = ref_secondary_network 
        else:
            self.ref_secondary_network = None

        if self.config.reflection_only:
            self.dif_network = None
            self.blend_network = None
        else:
            self.dif_network = dif_network
            self.blend_network = blend_network

    def diffuse(self, features):
        dif_network_inp = torch.cat([features.view(-1, features.shape[-1])], dim=-1)
        dif_color = self.dif_network(dif_network_inp).view(*features.shape[:-1], 3).float()
        if 'color_activation' in self.config:
            dif_color = get_activation(self.config.color_activation)(dif_color)
        return dif_color

    def reflective(self, features, dirs, *args):
        if self.config.dir_encoding_config.reflected:
            normals = args[0]
            dirs = -2 * torch.sum(dirs * normals, dim=-1, keepdim=True) * normals + dirs
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, dirs.shape[-1]))
        ref_network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        ref_color = self.ref_primary_network(ref_network_inp).view(*features.shape[:-1], 3).float()
        if 'color_activation' in self.config:
            ref_color = get_activation(self.config.color_activation)(ref_color)

        blend_network_inp = torch.cat([features.view(-1, features.shape[-1])], dim=-1)
        blend = self.blend_network(blend_network_inp).view(*features.shape[:-1], 1).float()
        if 'blend_activation' in self.config:
            blend = get_activation(self.config.blend_activation)(blend)

        return torch.cat([ref_color, blend], dim=-1)

    def secondary_shading(self, features, rays_o, rays_d, *args):
        rays_d = (rays_d + 1.) / 2. # (-1, 1) => (0, 1)
        xyz_embd = self.xyz_encoding(rays_o.view(-1, self.n_pos_dims))
        dirs_embd = self.dir_encoding(rays_d.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        # network_inp = torch.cat([xyz_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.ref_secondary_network(network_inp).view(*network_inp.shape[:-1], 4 if self.config.pred_vis else 3).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        visibility_smoothness = out['visibility_smoothness_loss_map'].mean()
        return {
            'visibility_smoothness': visibility_smoothness
        }


@models.register('volume-mix-radiance')
class VolumeMixRadiance(nn.Module):
    """
    Occlusion-aware volume radiance
    """
    def __init__(self, config):
        super(VolumeMixRadiance, self).__init__()
        self.config = config
        self.blend_type = self.config.get('blend_type', 'surface')
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_pos_dims = self.config.get('n_pos_dims', 3)
        self.n_output_dims = self.config.get('n_output_dims', 7)
        self.pred_vis = self.config.get('pred_vis', False)
        self.human_ref = self.config.get('human_ref', False)
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        xyz_encoding = get_encoding(self.n_pos_dims, self.config.xyz_encoding_config)
        ref_primary_network = get_mlp(
            self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims, 
            3, 
            self.config.ref_primary_network_config
        )
        ref_secondary_network = get_mlp(
            self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims + xyz_encoding.n_output_dims,
            # self.config.input_feature_dim + self.config.other_dim + dir_encoding.n_output_dims, 
            4 if self.config.pred_vis else 3, 
            self.config.ref_secondary_network_config
        )
        dif_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims,
            3, 
            self.config.diffuse_network_config
        )
        blend_network = get_mlp(
            self.config.input_feature_dim + xyz_encoding.n_output_dims,
            1, 
            self.config.blend_network_config
        )
        if self.human_ref:
            human_encoding = get_encoding(self.n_)
        self.dir_encoding = dir_encoding
        self.xyz_encoding = xyz_encoding
        self.ref_primary_network = ref_primary_network 

        if self.config.secondary_shading:
            self.ref_secondary_network = ref_secondary_network 
        else:
            self.ref_secondary_network = None

        if self.config.reflection_only:
            self.dif_network = None
            self.blend_network = None
        else:
            self.dif_network = dif_network
            self.blend_network = blend_network

    def forward(self, features, positions, dirs, *args):
        if self.config.dir_encoding_config.reflected:
            normals = args[0]
            dirs = -2 * torch.sum(dirs * normals, dim=-1, keepdim=True) * normals + dirs
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, dirs.shape[-1]))
        xyz_embd = self.xyz_encoding(positions.view(-1, positions.shape[-1]))
        ref_network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        dif_network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        blend_network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd], dim=-1)
        if self.config.blend_network_config.get("normlized_inp", False):
            blend_network_inp = F.normalize(blend_network_inp, dim=-1)

        ref_color = self.ref_primary_network(ref_network_inp).view(*features.shape[:-1], 3).float()
        if 'color_activation' in self.config:
            ref_color = get_activation(self.config.color_activation)(ref_color)
        if self.config.reflection_only:
            dif_color = torch.zeros((*features.shape[:-1], 3), device=dirs.device)
            blend = torch.zeros((*features.shape[:-1], 1), device=dirs.device)
        else:
            dif_color = self.dif_network(dif_network_inp).view(*features.shape[:-1], 3).float()
            blend = self.blend_network(blend_network_inp).view(*features.shape[:-1], 1).float()
            if 'color_activation' in self.config:
                dif_color = get_activation(self.config.color_activation)(dif_color)
            if 'blend_activation' in self.config:
                blend = get_activation(self.config.blend_activation)(blend)

        ret = torch.cat([dif_color, ref_color, blend], dim=-1)
        if self.blend_type == "volume":
            ret = torch.cat([ret, blend * dif_color + (1 - blend) * ref_color], dim=-1)
        return ret

    def secondary_shading(self, features, rays_o, rays_d, *args):
        rays_d = (rays_d + 1.) / 2. # (-1, 1) => (0, 1)
        xyz_embd = self.xyz_encoding(rays_o.view(-1, self.n_pos_dims))
        dirs_embd = self.dir_encoding(rays_d.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), xyz_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.ref_secondary_network(network_inp).view(*network_inp.shape[:-1], 4 if self.config.pred_vis else 3).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    # def secondary_shading(self, rays_o, rays_d, *args):
    #     rays_d = (rays_d + 1.) / 2. # (-1, 1) => (0, 1)
    #     dirs_embd = self.dir_encoding(rays_d.view(-1, self.n_dir_dims))
    #     pos_embd = self.xyz_encoding(rays_o.view(-1, self.n_pos_dims))
    #     network_inp = torch.cat([pos_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
    #     color = self.ref_secondary_network(network_inp).view(*network_inp.shape[:-1], 3).float()
    #     if 'color_activation' in self.config:
    #         color = get_activation(self.config.color_activation)(color)
    #     return color

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        visibility_smoothness = out['visibility_smoothness_loss_map'].mean()
        return {
            'visibility_smoothness': visibility_smoothness
        }

@models.register('oa-volume-radiance')
class OAVolumeRadiance(nn.Module):
    """
    Occlusion-aware volume radiance
    """
    def __init__(self, config):
        super(OAVolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_pos_dims = self.config.get('n_pos_dims', 3)
        self.n_output_dims = 3
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        xyz_encoding = get_encoding(self.n_pos_dims, self.config.xyz_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.dir_encoding = dir_encoding
        self.xyz_encoding = xyz_encoding
        self.network = network
        if self.config.secondary_shading:
            self.secondary_network = get_mlp(3 + xyz_encoding.n_output_dims + dir_encoding.n_output_dims, self.n_output_dims, self.config.secondary_mlp_network_config) # points, 

    def forward(self, features, dirs, *args):
        if self.config.dir_encoding_config.reflected:
            normals = args[0]
            dirs = -2 * torch.sum(dirs * normals, dim=-1, keepdim=True) * normals + dirs
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def secondary_shading(self, rays_o, rays_d, *args):
        if self.config.secondary_shading:
            rays_d = (rays_d + 1.) / 2. # (-1, 1) => (0, 1)
            dirs_embd = self.dir_encoding(rays_d.view(-1, self.n_dir_dims))
            pos_embd = self.xyz_encoding(rays_o.view(-1, self.n_pos_dims))
            network_inp = torch.cat([pos_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
            color = self.secondary_network(network_inp).view(*network_inp.shape[:-1], self.n_output_dims).float()
            if 'color_activation' in self.config:
                color = get_activation(self.config.color_activation)(color)
            return color
        else:
            return torch.zeros_like(rays_o, device=rays_o.device)

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        return {}

@models.register('volume-radiance-multi')
class VolumeRadiance_Multi(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance_Multi, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims + self.config.env_embedding.embd_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding = encoding
        self.network = network
        self.embedding = nn.Embedding(num_embeddings=self.config.env_embedding.n_embd, embedding_dim=self.config.env_embedding.embd_dim)

    def forward(self, features, dirs, light_idx, *args):
        if self.config.dir_encoding_config.reflected:
            normals = args[0]
            dirs = 2 * torch.sum(dirs * normals, dim=-1, keepdim=True) * normals - dirs
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        env_embd = self.embedding(light_idx)
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, env_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        self.n_other_dims = self.config.get('n_other_dims', 0)
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims + self.n_other_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding = encoding
        self.network = network

    def forward(self, features, dirs, *args):
        if self.config.dir_encoding_config.get("reflected", False):
            normals = args[0]
            dirs = 2 * torch.sum(dirs * normals, dim=-1, keepdim=True) * normals - dirs
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


@models.register("tensor-radiance")
class TensorRadiance(nn.Module):
    def __init__(self, config):
        super(TensorRadiance, self).__init__()
        self.config = config
        self.n_input_dims = self.config.get("n_input_dims", 3)
        self.n_dir_dims = self.config.get("n_dir_dims", 3)
        self.n_output_dims = 3
        self.radius = self.config['radius']
        # TensoRF uses additional PE to encode VM features
        xyz_encodings = []
        for conf in self.config.xyz_encoding_config:
            xyz_encodings.append(get_encoding(conf.args.n_input_dims, conf.args))
        xyz_encodings = nn.ModuleList(xyz_encodings)
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_feature_dims = (
            xyz_encodings[-1].n_output_dims
            + dir_encoding.n_output_dims
        )
        network = get_mlp(
            self.n_feature_dims, self.n_output_dims, self.config.mlp_network_config
        )
        self.xyz_encodings = xyz_encodings
        self.dir_encoding = dir_encoding
        self.network = network
        self.contract_to_unisphere = self.config.get("contract_to_unisphere", True)

    def forward(self, x, dirs, *args):
        if self.contract_to_unisphere:
            x = contract_to_unisphere(x, self.radius, self.contraction_type)
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        xyzs_embd = x
        for xyz_encoding in self.xyz_encodings:
            xyzs_embd = xyz_encoding(xyzs_embd)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat(
            [xyzs_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        color = (
            self.network(network_inp).view(*x.shape[:-1], self.n_output_dims).float()
        )
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        for xyz_encoding in self.xyz_encodings:
            update_module_step(xyz_encoding, epoch, global_step)
        update_module_step(self.dir_encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register("tensoir-appearance")
class TensoIRAppearance(nn.Module):
    def __init__(self, config):
        super(TensoIRAppearance, self).__init__()
        self.config = config
        self.n_input_dims = self.config.get("n_input_dims", 3)
        self.n_dir_dims = self.config.get("n_dir_dims", 3)
        self.n_output_dims = 3
        self.radius = self.config['radius']
        # TensoIR uses additional PE to encode VM features
        xyz_encodings = []
        for conf in self.config.xyz_encoding_config:
            xyz_encodings.append(get_encoding(conf.args.n_input_dims, conf.args))
        xyz_encodings = nn.ModuleList(xyz_encodings)
        pos_encoding = get_encoding(self.n_input_dims, self.config.pos_encoding_config)
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_rf_feature_dims = (
            xyz_encodings[-1].n_output_dims
            + dir_encoding.n_output_dims
        )
        self.n_intrinsic_feature_dims = (
            xyz_encodings[-1].n_output_dims
            + pos_encoding.n_output_dims
        )
        radiance_network = get_mlp(
            self.n_rf_feature_dims,
            self.config.radiance_mlp_network_config.n_output_dims,
            self.config.radiance_mlp_network_config,
        )
        normal_network = get_mlp(
            self.n_intrinsic_feature_dims,
            self.config.normal_mlp_network_config.n_output_dims,
            self.config.normal_mlp_network_config,
        )
        material_network = get_mlp(
            self.n_intrinsic_feature_dims,
            self.config.material_mlp_network_config.n_output_dims,
            self.config.material_mlp_network_config,
        )
        self.xyz_encodings = xyz_encodings
        self.pos_encoding = pos_encoding
        self.dir_encoding = dir_encoding
        self.radiance_network = radiance_network
        self.normal_network = normal_network
        self.material_network = material_network
        self.contract_to_unisphere = self.config.get("contract_to_unisphere", True)

    def forward(
        self, x, dirs, with_radiance=False, with_normal=False, with_mats=False, *args
    ):
        if self.contract_to_unisphere:
            x = contract_to_unisphere(x, self.radius, self.contraction_type)
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        xyzs_embd = x
        for xyz_encoding in self.xyz_encodings:
            xyzs_embd = xyz_encoding(xyzs_embd)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims))
        pos_embd = self.pos_encoding(x.view(-1, self.n_input_dims))
        rf_inp = torch.cat(
            [xyzs_embd, dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        intrinsic_inp = torch.cat(
            [xyzs_embd, pos_embd] + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        out = []
        if with_radiance:
            radiance = (
                self.radiance_network(rf_inp)
                .view(*x.shape[:-1], self.config.radiance_mlp_network_config.n_output_dims)
                .float()
            )
            out += [radiance]
        if with_normal:
            normal = (
                self.normal_network(intrinsic_inp)
                .view(*x.shape[:-1], self.config.normal_mlp_network_config.n_output_dims)
                .float()
            )
            out += [normal]
        if with_mats:
            material = (
                self.material_network(intrinsic_inp)
                .view(
                    *x.shape[:-1], self.config.material_mlp_network_config.n_output_dims
                )
                .float()
            )
            out += [material]

        return tuple(out) if len(out) > 1 else out[0]

    def get_radiance(self, feature):
        return self.radiance_network(feature).view(*feature.shape[:-1], self.config.radiance_mlp_network_config.n_output_dims).float()

    def get_normal(self, feature):
        return self.normal_network(feature).view(*feature.shape[:-1], self.config.normal_mlp_network_config.n_output_dims).float()

    def get_material(self, feature):
        return self.material_network(feature).view(*feature.shape[:-1], self.config.material_mlp_network_config.n_output_dims).float()

    def update_step(self, epoch, global_step):
        for xyz_encoding in self.xyz_encodings:
            update_module_step(xyz_encoding, epoch, global_step)
        update_module_step(self.dir_encoding, epoch, global_step)

    def regularizations(self, out):
        reg_dict = {}
        for xyz_encoding in self.xyz_encodings:
            if hasattr(xyz_encoding, 'regularizations'):
                for k, v in xyz_encoding.regularizations().items():
                    reg_dict[k + '_app'] = v

        return reg_dict
