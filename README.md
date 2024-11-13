# RISE-SDF: a Relightable Information-Shared Signed Distance Field for Glossy Object Inverse Rendering
## [Paper](https://www.arxiv.org/pdf/2409.20140)|[Project Page](https://dehezhang2.github.io/RISE-SDF/)|[Dataset](https://drive.google.com/drive/folders/1991eNN5-bMWK7aEHf99VU_iGZsH6FnAc?usp=drive_link)

​          ![teaser](./assets/teaser-1731247366796-2.gif)

This repository contains the implementation of our paper [RISE-SDF: a Relightable Information-Shared Signed Distance Field for Glossy Object Inverse Rendering](https://dehezhang2.github.io/RISE-SDF/).

You can find detailed usage instructions for training your own models below.

If you find our code useful, please cite:

```latex
@misc{zhang2024risesdfrelightableinformationsharedsigned,
	title={RISE-SDF: a Relightable Information-Shared Signed Distance Field for Glossy Object Inverse Rendering}, 
	author={Deheng Zhang and Jingyu Wang and Shaofei Wang and Marko Mihajlovic and Sergey Prokudin and Hendrik P. A. Lensch and Siyu Tang},
	year={2024},
	eprint={2409.20140},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	url={https://arxiv.org/abs/2409.20140}, 
}
```

## Requirements
**Note:**
- To utilize multiresolution hash encoding or fully fused networks provided by tiny-cuda-nn, you should have least an RTX 2080Ti, see [https://github.com/NVlabs/tiny-cuda-nn#requirements](https://github.com/NVlabs/tiny-cuda-nn#requirements) for more details.

### Install

```bash
git clone --recursive git@github.com:dehezhang2/RISE-SDF.git
```
### Environments

- Create `python 3.10` environment

  ```bash
  conda create --name instant-inv-glossy python=3.10
  conda activate rise-sdf
  ```

- Install PyTorch>=1.13 [here](https://pytorch.org/get-started/locally/) based the package management tool you used and your cuda version (older PyTorch versions may work but have not been tested)

  ```bash
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  ```

- Install Nvdiffrast

  ```bash
  git clone https://github.com/NVlabs/nvdiffrast.git
  cd nvdiffrast
  pip install .
  ```

- Install tiny-cuda-nn PyTorch extension and other dependencies (**make sure you are in the CUDA available mode**): 

  ```bash
  pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
  pip install -r requirements.txt
  ```

  - if you find a problem with `nerfacc`, uninstall your current nerfacc and build it again.
      ```bash
        pip uninstall nerfacc
        pip install git+https://github.com/nerfstudio-project/nerfacc.git
      ```

### Datasets & Preparation

* Download the [pre-integrated BSDF](https://github.com/liuyuan-pal/NeRO/blob/main/assets/bsdf_256_256.bin) into the `./load/bsdf/` folder.
* Download the environment maps from [TensorIR Dataset](https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view) into the `./load/` folder.
* Download [Shiny Inverse Rendering Dataset](https://drive.google.com/drive/folders/1991eNN5-bMWK7aEHf99VU_iGZsH6FnAc?usp=drive_link), save the files to the `./load/TensoIR_synthetic/ ` folder. 

## Physically Based Inverse Rendering (PBIR)

### Training

Run the launch script with `--train`, specifying the config file, the GPU(s) to be used (GPU 0 will be used by default), and the scene name:

```bash
# on Shiny Inverse Rendering Dataset
python launch.py --config configs/split-mixed-occ-tensoir.yaml --gpu 0 --train dataset.scene=toaster_disney 
```
The config snapshots, checkpoints, and experiment outputs are saved to `exp/[name]/[tag]@[timestamp]`. You can change any configuration in the YAML file by specifying arguments without `--`, for example:

```bash
python launch.py --config configs/split-mixed-occ-tensoir.yaml --gpu 0 --train dataset.scene=toaster_disney tag=iter50k seed=0 trainer.max_steps=50000
```

### Relighting (modify to per-scene script later)

The training procedure is by default followed by testing, which computes metrics on test data, generates animations, and exports the geometry as triangular meshes. If you want to do testing alone, just resume the pre-trained model and replace `--train` with `--test`, for example:

```bash
python launch.py --config exp/split-mixed-occ-tensoir-toaster_disney/your_experiment_directory/config/parsed.yaml --resume exp/split-mixed-occ-tensoir-toaster_disney/your_experiment_directory/ckpt/epoch\=0-step\=40000.ckpt --gpu 0 --test models.phys_kick_in_step=0
```

Note that you can change the relighting environment map easily in the config file

```yaml
dataset
  relight_list:  ['bridge', 'city']
  hdr_filepath: ./load/high_res_envmaps_2k/
```

## Acknowledgement

Our repo is developed based on [instant-nsr_pl](https://github.com/bennyguo/instant-nsr-pl), [IntrinsicAvatar](https://github.com/taconite/IntrinsicAvatar), [NeRO](https://github.com/liuyuan-pal/NeRO), [Nvdiffrec](https://github.com/NVlabs/nvdiffrec). Please also consider citing the corresponding papers. We thank authors of these papers for their wonderful works which greatly facilitates the development of our project.

## LICENSE

The code is released under the GPL-3.0 license.
