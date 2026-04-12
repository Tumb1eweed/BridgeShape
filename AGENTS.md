# Repository Guidelines

## Project Structure & Module Organization
Core training entry points live at the repository root: `train_vq.py` for VQ-VAE and `train_vqsdf.py` / `test_vqsdf.py` for the diffusion bridge pipeline. Model code is under `models/`, with reusable network blocks in `models/modules/` and VQ-specific code in `models/models_vq/`. Dataset loading lives in `datasets/` and `dataloaders/`. Config files are in `configs/`, category-specific shell launchers are in `scripts/`, and data preparation helpers are in `data/`.

## Build, Test, and Development Commands
Install Python dependencies with:
```bash
pip install -r requirements.txt
```
Prepare 3D-EPN data with:
```bash
python data/sdf_2_npy.py
python data/npy_2_pth.py
```
Train the Stage 1 VQ-VAE with a provided category script:
```bash
bash scripts/train_vqvae_snet_02933112.sh
```
Train the Stage 2 bridge model with a config:
```bash
python train_vqsdf.py --save_dir outputs/run1 --config configs/EPN_vqvae_res3d_chair.yaml
```
Run evaluation with:
```bash
python test_vqsdf.py --save_dir outputs/run1 --config configs/EPN_vqvae_res3d_chair.yaml --rs 1 --tbs 10 --v true
```

## Coding Style & Naming Conventions
Use 4-space indentation and follow existing Python style in the repository. Prefer `snake_case` for functions, variables, and file names; use `PascalCase` for classes. Keep config names descriptive and category-specific, for example `EPN_vqvae_res3d_chair.yaml`. No formatter or linter is currently configured, so match surrounding code closely and keep imports grouped and readable.

## Testing Guidelines
There is no dedicated unit-test suite yet. Validate changes by running the affected training or evaluation entry point with the relevant config. For data pipeline changes, smoke-test the loader path you touched. For model changes, record the exact command, config, and dataset split used.

## Commit & Pull Request Guidelines
Recent commits use short, direct subjects such as `Update args.py` and `Delete datasets/EPN_128.py`. Follow that style with imperative, focused commit messages. Keep each commit scoped to one change. Pull requests should include: purpose, affected configs or scripts, dataset assumptions, and any training or evaluation command used to verify the change. Include sample outputs or screenshots only when visualization behavior changes.

## Configuration & Data Notes
Do not hardcode local dataset paths or GPU IDs in committed code. Keep large datasets and checkpoints out of Git. Store generated outputs under a dedicated run directory such as `outputs/` or `ckps/`.
