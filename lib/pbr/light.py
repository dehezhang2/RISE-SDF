import numpy as np
import torch
import torch.nn.functional as F
import tinycudann as tcnn

# import models

from .utils import nvdiffrecmc_util as util
from .utils.light_utils import compute_energy, fibonacci_sphere, eval_SGs, cubemap_mip, cubemap_to_blender_latlong, blender_latlong_to_cubemap,  cubemap_to_nmf_latlong, nmf_latlong_to_cubemap, cubemap_to_latlong
import nvdiffrast.torch as dr
from lib import renderutils as ru

######################################################################################
# Monte-carlo sampled environment light with PDF / CDF computation
######################################################################################


class EnvironmentLightBase(torch.nn.Module):
    def __init__(self, config):
        super(EnvironmentLightBase, self).__init__()

    def parameters(self):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    # def clone(self):
    #     raise NotImplementedError("EnvironmentLightBase is an abstract class")

    # def clamp_(self, min=None, max=None):
    #     raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def pdf(self, directions):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    def eval(self, directions):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def sample(self, num_samples: int):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def update_pdf(self):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def generate_image(self):
        raise NotImplementedError("EnvironmentLightBase is an abstract class")

    @torch.no_grad()
    def sample_stratified(
        self,
        batch_size: int,
        n_rows: int,
        n_cols: int,
        device: torch.device,
    ):
        """
        Stratified sampling of the environment map. Modified from TensoIR's implementation.
        Args:
            batch_size: The number of batches (pixels) to sample
            n_rows: The number of rows for each batch
            n_cols: The number of columns for each batch
            device: The device to put the sampled directions on
            shuffle: Whether to shuffle the batches
        Returns:
            A tuple (indices, pdfs) where:
                indices: A tensor of shape (num_samples, 2) containing sampled row and column indices
                pdfs: A tensor of shape (num_samples,) containing the pdf values of the sampled indices
        """
        lat_step_size = np.pi / n_rows
        lng_step_size = 2 * np.pi / n_cols

        # Generate theta in [pi/2, -pi/2] and phi in [pi, -pi]
        theta, phi = torch.meshgrid(
            [
                torch.linspace(
                    np.pi / 2 - 0.5 * lat_step_size,
                    -np.pi / 2 + 0.5 * lat_step_size,
                    n_rows,
                    device=device,
                ),
                torch.linspace(
                    np.pi - 0.5 * lng_step_size,
                    -np.pi + 0.5 * lng_step_size,
                    n_cols,
                    device=device,
                ),
            ],
            indexing="ij",
        )
        sin_theta = torch.sin(
            torch.pi / 2 - theta
        )  # convert theta from [pi/2, -pi/2] to [0, pi]
        inv_pdf = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [H, W]
        inv_pdf = inv_pdf.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, H, W]
        if self.training:
            phi_jitter = lng_step_size * (
                torch.rand(batch_size, n_rows, n_cols, device=device) - 0.5
            )
            theta_jitter = lat_step_size * (
                torch.rand(batch_size, n_rows, n_cols, device=device) - 0.5
            )

            theta, phi = theta[None, ...] + theta_jitter, phi[None, ...] + phi_jitter

        directions = torch.stack(
            [
                torch.cos(phi) * torch.cos(theta),
                torch.sin(phi) * torch.cos(theta),
                torch.sin(theta),
            ],
            dim=-1,
        )  # training: [B, H, W, 3], testing: [H, W, 3]
        directions = F.normalize(directions, dim=-1)
        if not self.training:
            directions = directions.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Final return: [B*H*W, 3], [B*H*W, 1]
        return directions.reshape(-1, 3), inv_pdf.reshape(-1, 1)




# TODO: there should be a way to analytically derived the sample() and pdf() functions
class EnvironmentLightMipCube(EnvironmentLightBase):
    LIGHT_MIN_RES = 16
    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5
    
    def __init__(self, config):
        super(EnvironmentLightMipCube, self).__init__(config)
        # self.mtx = None
        scale = config.envlight_config.scale
        bias = config.envlight_config.bias
        base_res = config.envlight_config.base_res
        
        if config.envlight_config.hdr_filepath is None:
            base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
        else:
            self.hdr_filepath = config.envlight_config.hdr_filepath
            latlong_img = torch.tensor(util.load_image(config.envlight_config.hdr_filepath).copy(), dtype=torch.float32, device='cuda')
            if config.envlight_config.clamp:
                latlong_img = latlong_img.clamp(0, 1)
            if not config.envlight_config.nmf_format:
                base = blender_latlong_to_cubemap(latlong_img, [512, 512])
            else:
                base = nmf_latlong_to_cubemap(latlong_img, [512, 512])

        
        self.register_parameter("base", torch.nn.Parameter(base))
        self.config = config
        # self.pdf_scale = (self.base.shape[0] * self.base.shape[1] * self.base.shape[2]) / (2 * np.pi * np.pi)
        # self.update_pdf()
    def relight(self, file_name):
        latlong_img = torch.tensor(util.load_image(file_name), dtype=torch.float32, device='cuda')
        base = blender_latlong_to_cubemap(latlong_img, [512, 512]) if not self.config.envlight_config.nmf_format else nmf_latlong_to_cubemap(latlong_img, [512, 512])
        self.register_parameter("base", torch.nn.Parameter(base))

    def freeze(self):
        self.base.requires_grad = False
        self.base = self.base.detach_()
        self.diffuse = self.diffuse.detach_()
        new_specular = []
        for mip in self.specular:
            new_specular.append(mip.detach_())
        self.specular = new_specular

    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]    
        self.diffuse = ru.diffuse_cubemap(self.specular[-1])
        

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)

    # query the mip map in stage one
    def eval_mip(self, directions, specular=False, roughness=None):
        pn, bn = directions.shape[0], 1
        if specular:
            assert roughness != None 
            miplevel = self.get_mip(roughness)
            miplevel = miplevel.reshape(1, pn//bn, bn, miplevel.shape[-1])
            direct_light = dr.texture(self.specular[0][None, ...], 
                                  directions.reshape(1, pn // bn, bn, directions.shape[-1]).contiguous(),
                                  mip=list(m[None, ...] for m in self.specular[1:]), 
                                  mip_level_bias=miplevel[..., 0], 
                                  filter_mode='linear-mipmap-linear', 
                                  boundary_mode='cube')
            return direct_light.reshape(direct_light.shape[1], -1)
        else:
            light = dr.texture(self.diffuse[None, ...], 
                           directions.reshape(1, pn // bn, bn, directions.shape[-1]).contiguous(), 
                           filter_mode='linear', 
                           boundary_mode='cube')
            return light.reshape(light.shape[1], -1)
        
            
    def parameters(self):
        return [self.base]

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    @torch.no_grad()
    def pdf(self, directions):
        """
        Compute the PDFs of the given directions based on the environment map
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N,) containing the PDFs for each input direction
        """
        # Convert the 3D directions to 2D indices in the environment map
        phi = torch.atan2(directions[:, 1], directions[:, 0])  # Compute azimuth angle
        theta = torch.acos(directions[:, 2])  # Compute elevation angle
        u = (phi + np.pi) / (2 * np.pi)  # Map azimuth to [0, 1]
        v = theta / np.pi  # Map elevation to [0, 1]

        # Convert u, v to discrete indices
        col_indices = torch.clamp(
            torch.floor(u * (self.cols.shape[1] - 1)), min=0, max=self.cols.shape[1] - 2
        )
        row_indices = torch.clamp(
            torch.floor(v * (self.rows.shape[0] - 1)), min=0, max=self.rows.shape[0] - 2
        )

        # Get PDF values at the indices
        sin_theta = torch.sin(theta)
        pdf_values = torch.where(
            sin_theta > 0,
            self._pdf[row_indices.long(), col_indices.long()]
            * self.pdf_scale
            / sin_theta,
            torch.zeros_like(sin_theta),
        )

        return pdf_values.unsqueeze(-1)

    def eval(self, directions):
        """
        Evaluate the environment light intensities at the given directions
        Args:
            directions: A tensor of shape (N, 3) containing unit vectors
        Returns:
            A tensor of shape (N, C) containing the environment light intensities at the input directions
        """
        pn, bn = directions.shape[0], 1
        light = dr.texture(self.base[None, ...], 
                           directions.reshape(1, pn // bn, bn, directions.shape[-1]).contiguous(), 
                           filter_mode='linear', 
                           boundary_mode='cube')
        return light.reshape(light.shape[1], -1)

    @torch.no_grad()
    def sample(self, num_samples: int):
        """
        Importance sample continuous locations on the environment light based on discrete CDFs
        Args:
            num_samples: Number of samples to generate
        Returns:
            A tuple (indices, pdfs) where:
                indices: A tensor of shape (num_samples, 2) containing sampled row and column indices
                pdfs: A tensor of shape (num_samples,) containing the pdf values of the sampled indices
        """
        # Generate random numbers for rows and columns
        u1 = torch.rand(num_samples, device=self.base.device)
        u2 = (
            torch.rand(num_samples, device=self.base.device).reshape(-1, 1).contiguous()
        )

        # Find the row indices based on the random numbers u1 and the row CDF
        # TODO: check for divide-by-zero cases - probably not needed
        row_indices = torch.searchsorted(self.rows[:, 0].contiguous(), u1, right=True)
        below = torch.max(torch.zeros_like(row_indices - 1), row_indices - 1)
        above = torch.min(
            (self.rows.shape[0] - 1) * torch.ones_like(row_indices), row_indices
        )
        row_fracs = (u1 - self.rows[below, 0]) / (
            self.rows[above, 0] - self.rows[below, 0]
        )
        row_indices = below

        # For each row index, find the column index based on the random numbers u2 and the column CDF
        # Use advanced indexing to vectorize the operation
        col_indices = torch.searchsorted(
            self.cols[row_indices], u2, right=True
        ).squeeze(-1)
        below = torch.max(torch.zeros_like(col_indices - 1), col_indices - 1)
        above = torch.min(
            (self.cols.shape[-1] - 1) * torch.ones_like(col_indices), col_indices
        )
        col_fracs = (u2.squeeze(-1) - self.cols[row_indices, below]) / (
            self.cols[row_indices, above] - self.cols[row_indices, below]
        )
        col_indices = below

        # Concatenate the row and column indices to get a 2D index for each sample
        # Add the fractions to get continuous coordinates
        uv = torch.stack(
            [
                (col_indices + col_fracs) / self.base.shape[1],
                (row_indices + row_fracs) / self.base.shape[0],
            ],
            dim=1,
        )

        # Convert the 2D indices to spherical coordinates
        theta = uv[:, 1] * np.pi
        phi = uv[:, 0] * np.pi * 2 - np.pi

        # Convert spherical coordinates to directions
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        directions = torch.stack(
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=1
        )
        directions = F.normalize(directions, dim=1)

        return directions

    @torch.no_grad()
    def update_pdf(self):
        # Compute PDF
        Y = util.pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
        self._pdf = torch.max(self.base, dim=-1)[0] * torch.sin(
            Y * np.pi
        )  # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
        self._pdf[self._pdf <= 0] = 1e-6  # avoid divide by zero in sample()
        self._pdf = self._pdf / torch.sum(self._pdf)  # discrete pdf

        # Compute cumulative sums over the columns and rows
        self.cols = torch.cumsum(self._pdf, dim=1)
        self.rows = torch.cumsum(
            self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0
        )

        # Normalize
        # TODO: for columns/rows with all 0s, use uniform distribution
        self.cols = self.cols / torch.where(
            self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols)
        )
        self.rows = self.rows / torch.where(
            self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows)
        )

        # Prepend 0s to all CDFs
        self.cols = torch.cat([torch.zeros_like(self.cols[:, :1]), self.cols], dim=1)
        self.rows = torch.cat([torch.zeros_like(self.rows[:1, :]), self.rows], dim=0)

        # self._pdf *= self.base.shape[1] * self.base.shape[0]

    @torch.no_grad()
    def generate_image(self):
        # return self.base.detach().cpu().numpy()
        color = cubemap_to_latlong(self.base, [512, 1024]) if not self.config.envlight_config.nmf_format else cubemap_to_nmf_latlong(self.base, [512, 1024])
        return color

