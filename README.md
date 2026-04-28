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

All training, evaluation, data-preparation, and smoke-test commands in this repository should be run from the `bridgeshape` Conda environment:

```bash
conda activate bridgeshape
```

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

On the current machine (`RTX 4080 32GB`, single GPU), the `02691156` Stage 1 script has been smoke-tested with `batch_size=48`. `batch_size=52` and above triggered CUDA OOM in the first few iterations, so `48` is the current single-GPU recommendation.

### Stage 2: Train Latent Diffusion Schrödinger Bridge
Once the VQ-VAE is trained, you can train the diffusion bridge model using the configuration files located in the ```configs/``` directory.
```angular2html
CUDA_VISIBLE_DEVICES=0 python train_vqsdf.py \
    --save_dir [YOUR_SAVE_DIRECTORY] \
    --config configs/[YOUR_CONFIG].yaml
```

For the `02691156` plane configuration on the current machine, `configs/EPN_vqvae_res3d_plane.yaml` has been validated with single-GPU `training.bs=62` and `evaluation.bs=31`. `training.bs=64` caused CUDA OOM during backpropagation, so `62` is the current single-GPU recommendation.

### One-Epoch Smoke Test for Network Changes
When changing the VQ-VAE, diffusion bridge, or shared network blocks, use the `02691156` airplane/plane category as a quick end-to-end check. This verifies that Stage 1 training, Stage 2 training, checkpoint loading, and Stage 2 evaluation all run before starting a full experiment.

The `02691156` split contains `19,800` training samples and `4,470` test samples. On the current single-GPU setup, one data epoch is:

* Stage 1 VQ-VAE: `ceil(19800 / 48) = 413` steps.
* Stage 2 bridge: `ceil(19800 / 62) = 320` steps.

Run Stage 1 for one epoch and save a checkpoint at the final step:

```bash
conda activate bridgeshape
cd /root/autodl-tmp/BridgeShape/scripts
OMP_NUM_THREADS=1 \
BATCH_SIZE=48 \
TOTAL_ITERS=413 \
SAVE_STEPS_FREQ=413 \
PRINT_FREQ=50 \
DISPLAY_FREQ=250000 \
bash train_vqvae_snet_02691156.sh
```

Expected Stage 1 checkpoint:

```text
/root/autodl-tmp/BridgeShape/logs_home_32/vqvae-ControlledEPNDataset_32-02691156plane-all-res32-LR1e-4-T3.0-release/ckpt/vqvae_steps-413.pth
```

Run Stage 2 for one epoch with the smoke-test config:

```bash
cd /root/autodl-tmp/BridgeShape
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train_vqsdf.py \
    --save_dir outputs/one_epoch_02691156 \
    --config configs/EPN_vqvae_res3d_plane_stage2_1epoch.yaml \
    --name EPN_vqvae_res3d_plane_stage2_1epoch
```

Expected Stage 2 checkpoint:

```text
outputs/one_epoch_02691156/EPN_vqvae_res3d_plane_stage2_1epoch/step_320.pth
```

Evaluate the Stage 2 checkpoint on the full `02691156` test split:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python test_vqsdf.py \
    --save_dir outputs/one_epoch_02691156 \
    --config configs/EPN_vqvae_res3d_plane_stage2_1epoch.yaml \
    --name EPN_vqvae_res3d_plane_stage2_1epoch \
    --rs 1 \
    --tbs 10 \
    --test_start_epoch 7 \
    --test_one true
```

The `--test_start_epoch 7` value is chosen so that the current test script resolves to `step_320.pth` with `training.log_interval=50` and `training.save_interval=320`. Results are appended to:

```text
outputs/one_epoch_02691156/1_steps/result.txt
```

This smoke test is intended to catch runtime breakage, tensor shape mismatches, checkpoint incompatibilities, and obvious CUDA memory issues. A one-epoch run is not expected to produce final-quality metrics.

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
