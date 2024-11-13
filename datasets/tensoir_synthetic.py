import os
import json
import math
import pyexr
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from lib.pbr import rgb_to_srgb, srgb_to_rgb


class TensoIRDatasetBase:
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True


        self.root_dir = Path(self.config.root_dir)
        self.split_list = [x for x in self.root_dir.iterdir() if x.stem.startswith(self.split)]
        self.split_list.sort()




        with open(
            os.path.join(self.config.root_dir, f"{self.split}_000/metadata.json"), "r"
        ) as f:
            meta = json.load(f)

        if "w" in meta and "h" in meta:
            W, H = int(meta["w"]), int(meta["h"])
        else:
            W, H = 800, 800

        if "img_wh" in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif "img_downscale" in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        # near 2.0 far 6.0
        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = (
            0.5 * w / math.tan(0.5 * meta["cam_angle_x"])
        )  # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            self.w,
            self.h,
            self.focal,
            self.focal,
            self.w // 2,
            self.h // 2,
            openGL_camera=config.openGL_camera,
        ).to(
            self.rank
        )  # (h, w, 3)

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.all_normals = []
        if self.config.has_albedo:
            self.all_albedo = []

        if self.config.has_roughness:
            self.all_roughness = []

        self.relight_images = {}
        for light in self.config.relight_list:
            self.relight_images[light] = []

        for idx in tqdm(range(len(self.split_list))):
            item_path = self.split_list[idx]
            item_meta_path = item_path / 'metadata.json'
            with open(item_meta_path, 'r') as f:
                meta = json.load(f)
            c2w = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
            c2w = torch.FloatTensor(c2w[:3, :4])
            self.all_c2w.append(c2w)
            img_path = item_path / f'rgba.png'
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
            self.all_fg_masks.append(img[..., -1])  # (h, w)
            self.all_images.append(img[..., :3])

            if self.config.has_albedo:
                albedo_path = item_path / ('albedo.' + self.config.albedo_format)
                if self.config.albedo_format=='exr':
                    albedo_img = pyexr.open(str(albedo_path)).get()
                    albedo_img = torch.from_numpy(albedo_img).float()
                else:
                    albedo_img = Image.open(albedo_path)
                    albedo_img = albedo_img.resize(self.img_wh, Image.BICUBIC)
                    albedo_img = TF.to_tensor(albedo_img).permute(1, 2, 0)

                self.all_albedo.append(albedo_img[..., :3])

            if self.config.has_roughness:
                roughness_path = item_path / ('roughness.' + self.config.roughness_format)
                if self.config.albedo_format=='exr':
                    roughness_img = pyexr.open(str(roughness_path)).get()
                    roughness_img = torch.from_numpy(roughness_img).float()
                else:
                    roughness_img = Image.open(roughness_path)
                    roughness_path = roughness_path.resize(self.img_wh, Image.BICUBIC)
                    roughness_path = TF.to_tensor(roughness_path).permute(1, 2, 0)

                self.all_roughness.append(roughness_img[..., :1])

            # use original normal value
            normal_path = item_path / f'normal.exr'
            normal_img = pyexr.open(str(normal_path)).get()
            
            normal_bg = np.array([0.0, 0.0, 1.0])
            normal_alpha = normal_img[..., [-1]]
            normal_img = normal_img[..., :3]
            normal_img = normal_img * normal_alpha + normal_bg * (1.0 - normal_alpha)  # [H, W, 3]
            normal_img = torch.from_numpy(normal_img).float()
           
            normal_img = normal_img / torch.norm(normal_img, dim=-1, keepdim=True)
            self.all_normals.append(normal_img)


            for light in self.config.relight_list:
                relight_img_path = item_path / f'rgba_{light}.png'
                relight_img = Image.open(relight_img_path)
                relight_img = relight_img.resize(self.img_wh, Image.BICUBIC)
                relight_img = TF.to_tensor(relight_img).permute(1, 2, 0).to(self.rank)
                self.relight_images[light].append(relight_img[..., :3])

            
        for light in self.config.relight_list:
            self.relight_images[light] = torch.stack(self.relight_images[light], dim=0).float()

        self.all_c2w, self.all_images, self.all_fg_masks = (
            torch.stack(self.all_c2w, dim=0).float().to(self.rank),
            torch.stack(self.all_images, dim=0).float().to(self.rank),
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank),
        )
        if self.config.has_albedo:
            self.all_albedo = torch.stack(self.all_albedo, dim=0).float()
        if self.config.has_roughness:
            self.all_roughness = torch.stack(self.all_roughness, dim=0).float()
        self.all_normals = torch.stack(self.all_normals, dim=0).float()



class TensoIRDataset(Dataset, TensoIRDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class TensoIRIterableDataset(IterableDataset, TensoIRDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register("tensoir")
class TensoIRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = TensoIRIterableDataset(
                self.config, self.config.train_split
            )
        if stage in [None, "fit", "validate"]:
            self.val_dataset = TensoIRDataset(self.config, self.config.val_split)
        if stage in [None, "test"]:
            self.test_dataset = TensoIRDataset(self.config, self.config.test_split)
        if stage in [None, "predict"]:
            self.predict_dataset = TensoIRDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=8,
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler,
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
