# BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion [AAAI 2026]
## [Paper](https://arxiv.org/abs/2506.23205)

Official implementation of the paper **"BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion"**.


---

## 📢 News & Updates
* **[March 2026]** Initial release of the core BridgeShape framework (Latent Diffusion Schrödinger Bridge). 
* **[Upcoming]** Release of the Depth-Enhanced VQ-VAE module featuring the self-projected Multi-View depth feature. Stay tuned!

---

## 🛠️ Environments

The verified environment for this repository is a Conda environment named `bridgeshape` with `Python 3.10`, `PyTorch 2.4.1`, `CUDA 12.1`, and the prebuilt `PyTorch3D 0.7.8` package.

Create the environment with:

```bash
conda create -n bridgeshape -y python=3.10
conda install -n bridgeshape -y -c pytorch3d -c pytorch -c nvidia -c conda-forge \
    pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 \
    pytorch-cuda=12.1 pytorch3d=0.7.8 iopath fvcore
conda activate bridgeshape
pip install -r requirements.txt
```

This setup has been validated on the current codebase. We do not recommend using `Python 3.12` with the latest default `torch` packages here, because `pytorch3d` binary compatibility becomes unreliable and often falls back to a local source build.

The provided `environment.yml` matches this validated environment and can also be used directly:

```bash
conda env create -f environment.yml
conda activate bridgeshape
```

## 📂 Data Construction

We utilize the 3D-EPN dataset for our experiments. Please download the original data available from the [3D-EPN](https://graphics.stanford.edu/projects/cnncomplete/data.html) project page for both training and evaluation.

To run the default setting with a resolution of 32³, download the necessary data files [shapenet_dim32_df.zip](http://kaldir.vc.in.tum.de/adai/CNNComplete/shapenet_dim32_df.zip) (completed shapes) and [shapenet_dim32_sdf.zip](http://kaldir.vc.in.tum.de/adai/CNNComplete/shapenet_dim32_sdf.zip) (partial shapes).

To prepare the data:

Run `data/sdf_2_npy.py` to convert the raw files into `.npy` format for easier handling.

Run `data/npy_2_pth.py` to generate the paired data for the eight object classes used for model training.

The preprocessing scripts are aligned to the available partial set in `shapenet_dim32_sdf`:

```bash
conda activate bridgeshape
python data/sdf_2_npy.py
python data/npy_2_pth.py
```

Expected layout on this machine:

```text
/root/autodl-tmp/datasets
├── shapenet_dim32_df
├── shapenet_dim32_sdf
└── control_data
```

Important notes:

* `data/sdf_2_npy.py` only converts complete-shape `df` files that are actually referenced by the available partial `sdf` files. It does **not** process the full `shapenet_dim32_df` archive.
* Each generated `.pth` in `control_data` stores a `(partial, complete)` pair as `torch.save((sdf, df), out_file)`.
* The `partial` tensor comes from `shapenet_dim32_sdf`, while the `complete` tensor comes from the matched `shapenet_dim32_df` sample.
* After `control_data` has been generated, the intermediate `shapenet_dim32_df_npy` and `shapenet_dim32_sdf_npy` directories can be deleted to reclaim disk space.

Your data structure should be organized as follows before starting the training process:

```
/root/autodl-tmp/datasets
├── control_data
│   ├── 02691156
│   │   ├── 10155655850468db78d106ce0a280f87__0__.pth
│   │   ├── ...
│   ├── 02933112
│   ├── 03001627
│   ├── ...
│   ├── splits
│   │   ├── train_02691156.txt
│   │   ├── train_02933112.txt
│   │   ├── ...
│   │   ├── test_02691156.txt
│   │   ├── test_02933112.txt
│   │   ├── ...
```

## 🚀 Training
BridgeShape employs a two-stage training pipeline. We train category-specific models for the eight distinct categories.

### Stage 1: Train & Test VQ-VAE
First, train the VQ-VAE to establish the compact latent space. We provide shell scripts for each of the eight categories in the ```scripts/``` directory.

Simply run the corresponding ```.sh``` file for your desired category:
```angular2html
bash scripts/train_vqvae_snet_*.sh
```
Note: The script handles both training and testing. The best performing weights will be automatically saved to ```ckps/vqvae_epoch-best.pth```.

### Stage 2: Train Latent Diffusion Schrödinger Bridge
Once the VQ-VAE is trained, you can train the diffusion bridge model using the configuration files located in the ```configs/``` directory.
```angular2html
CUDA_VISIBLE_DEVICES=0 python train_vqsdf.py \
    --save_dir [YOUR_SAVE_DIRECTORY] \
    --config configs/[YOUR_CONFIG].yaml
```

## 🔍 Inference / Testing
To evaluate your trained BridgeShape model, run the testing script with your desired sampling configurations:
```angular2html
CUDA_VISIBLE_DEVICES=0 python test_vqsdf.py \
    --save_dir [YOUR_SAVE_DIRECTORY] \
    --config configs/[YOUR_CONFIG].yaml \
    --rs 1 \
    --tbs 10 \
    --test_start_epoch 5 \
    --test_one false \
    --v true
```

**Testing Arguments:**
* `--rs`: Number of reverse sampling steps.
* `--tbs`: Test batch size.
* `--test_start_epoch`: The epoch number from which to start testing.
* `--test_one`: Set to `true` to run only a single evaluation round.
* `--v`: Set to `true` to save the output visualization results.

## 📖 Citation
If you find our work or code useful for your research, please consider citing our paper:
```
@inproceedings{kong2026bridgeshape,
  title={BridgeShape: Latent Diffusion Schr{\"o}dinger Bridge for 3D Shape Completion},
  author={Kong, Dequan and Chen, Honghua and Zhu, Zhe and Wei, Mingqiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={7},
  pages={5726--5734},
  year={2026}
}
```

## 🙏 Acknowledgement
Our implementation builds upon several excellent open-source projects. We express our gratitude to the authors of:

 [DiffComplete](https://github.com/JIA-Lab-research/DiffComplete) for structural inspiration.

 [P2P-Bridge](https://github.com/matvogel/P2P-Bridge) for insights into diffusion bridges.
