import gc
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn


def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)

    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=sigma.device, dtype=sigma.dtype) + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x

def match_colors_for_image_set_hard_mask(rendered_img, edited_img, image_mask):
    """
    rendered_img: [H, W, 3]
    edited_img: [H, W, 3]
    image_mask: [H, W]
    """
    sh = rendered_img.shape

    rendered_img = rendered_img.view(-1, 3)
    edited_img = edited_img.view(-1, 3).to(rendered_img.device)
    image_mask = image_mask.view(-1).to(rendered_img.device)

    labels = torch.unique(image_mask).to(rendered_img.device)

    for label in labels:
        image_region = rendered_img[image_mask==label]
        style_region = edited_img[image_mask==label]
        mu_c = image_region.mean(0, keepdim=True)
        mu_s = style_region.mean(0, keepdim=True)
        cov_c = torch.matmul((image_region - mu_c).transpose(1, 0), image_region - mu_c) / float(image_region.size(0))
        cov_s = torch.matmul((style_region - mu_s).transpose(1, 0), style_region - mu_s) / float(style_region.size(0))
        
        u_c, sig_c, _ = torch.svd(cov_c)
        u_s, sig_s, _ = torch.svd(cov_s)
        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_region = image_region @ tmp_mat.T + tmp_vec.view(1, 3)
        image_region = image_region.contiguous().clamp_(0.0, 1.0)
        rendered_img[image_mask==label] = image_region
    rendered_img = rendered_img.view(sh)
    return rendered_img

def match_colors_for_image_set_hard_mask_albedo(rendered_img, edited_img, rendered_albedo, image_mask):
    """
    rendered_img: [H, W, 3]
    edited_img: [H, W, 3]
    rendered_albedo: [H, W, 3]
    image_mask: [H, W]
    """
    sh = rendered_img.shape
    assert sh == rendered_albedo.shape
    rendered_img = rendered_img.view(-1, 3)
    rendered_albedo = rendered_albedo.view(-1, 3)
    edited_img = edited_img.view(-1, 3).to(rendered_img.device)
    image_mask = image_mask.view(-1).to(rendered_img.device)

    labels = torch.unique(image_mask).to(rendered_img.device)

    for label in labels:
        image_region = rendered_img[image_mask==label]
        albedo_region = rendered_albedo[image_mask==label]
        style_region = edited_img[image_mask==label]
        mu_c = image_region.mean(0, keepdim=True)
        mu_s = style_region.mean(0, keepdim=True)
        cov_c = torch.matmul((image_region - mu_c).transpose(1, 0), image_region - mu_c) / float(image_region.size(0))
        cov_s = torch.matmul((style_region - mu_s).transpose(1, 0), style_region - mu_s) / float(style_region.size(0))
        
        u_c, sig_c, _ = torch.svd(cov_c)
        u_s, sig_s, _ = torch.svd(cov_s)
        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_region = image_region @ tmp_mat.T + tmp_vec.view(1, 3)
        image_region = image_region.contiguous().clamp_(0.0, 1.0)

        albedo_region = albedo_region @ tmp_mat.T + tmp_vec.view(1, 3)
        albedo_region = albedo_region.contiguous().clamp_(0.0, 1.0)

        rendered_img[image_mask==label] = image_region
        rendered_albedo[image_mask==label] = albedo_region

    rendered_img = rendered_img.view(sh)
    rendered_albedo = rendered_img.view(sh)
    return rendered_img, rendered_albedo