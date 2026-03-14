# Recon FID V2 Runbook (Deterministic Reconstruction)

This runbook defines the updated reconstruction-first approach to improve VAE reconstruction FID.

## Class mapping

- `0` - CNV
- `1` - DME
- `2` - DRUSEN
- `3` - NORMAL

## Server session setup (on5/on4)

### 1) SSH and go to repository

```bash
ssh mhuber@on5.cir.meduniwien.ac.at
cd /optima/exchange/mhuber/new_git/maisi/
```

### 2) Start tmux (or attach if it already exists)

If you get `duplicate session: <name>`, the session already exists.

```bash
tmux attach -t maisi_vae_128
```

To create a brand-new session with a new name:

```bash
tmux new -s maisi_vae_128_v2
```

Optional helper commands:

```bash
tmux ls
tmux kill-session -t maisi_vae_128
```

### 3) Request a GPU node

on5:

```bash
srun -n16 --mem=50G --qos=longrunning --time=12-12:00:00 --gres=gpu:1 --nodelist=on5 -p full_optima -J "vae_reconfid_v2" --pty /bin/bash
```

on4 fallback:

```bash
srun -n16 --mem=50G --qos=longrunning --time=12-12:00:00 --gres=gpu:1 --nodelist=on4 -J "vae_reconfid_v2" --pty /bin/bash
```

### 4) Build/enter Singularity

Build once (only if `maisi.sif` is missing/outdated):

```bash
sudo singularity build maisi.sif maisi.def
```

Enter container:

```bash
singularity shell --nv maisi.sif
```

## What changed

- New config: `configs/config_VAE_full128_reconfid_v2.json`
- No train-time augmentation/noise
- Lower adversarial pressure (`adv_weight=0.005`)
- Full precision training (`amp=false`, `norm_float16=false`)
- Full validation per eval (`max_val_steps=null`)
- Best-checkpoint-only saving (`save_best_only=true`)
- Deterministic reconstruction export (`--deterministic`)
- Full precision latent export (no CUDA autocast in `scripts/encode_to_latents.py`)

## 1) Train VAE with the new config

```bash
python train_vae.py --config ./configs/config_VAE_full128_reconfid_v2.json
```

Training writes to `./runs/VAE/vae_full128_reconfid_v2_<timestamp>/` and also creates a stable pointer:

- `./runs/VAE/vae_full128_reconfid_v2_best.pt`

## 2) Build reconstruction dataset for recon-FID

Use deterministic decode from `z_mu`.

```bash
python ./scripts/build_vae_recon_dataset.py \
  --real_root "/optima/exchange/mhuber/KermanyV3_resized/train" \
  --out_root "/optima/exchange/mhuber/pw/outputs/recon_full128_reconfid_v2" \
  --vae_ckpt "./runs/VAE/vae_full128_reconfid_v2_best.pt" \
  --deterministic \
  --device cuda
```

## 3) Evaluate with standard (Inception) FID

```bash
python ./compute_oct_metrics.py \
  --real_root "/optima/exchange/mhuber/KermanyV3_resized/train" \
  --fake_root "/optima/exchange/mhuber/pw/outputs/recon_full128_reconfid_v2" \
  --feature_extractor inception-v3-compat \
  --seed 42 \
  --out_dir "./metrics_reports"
```

## 4) Evaluate with OCT-domain (RETFound) FID

```bash
python ./compute_oct_metrics.py \
  --real_root "/optima/exchange/mhuber/KermanyV3_resized/train" \
  --fake_root "/optima/exchange/mhuber/pw/outputs/recon_full128_reconfid_v2" \
  --feature_extractor retfound-mae \
  --feature_extractor_weights_path "/optima/exchange/mhuber/pw/checkpoints/RETFound_mae_natureOCT.pth" \
  --seed 42 \
  --out_dir "./metrics_reports"
```

## Where NOT to add `--half`

Do not add `--half` when running `scripts/build_vae_recon_dataset.py` for recon-FID evaluation. Keep reconstruction export in full precision to avoid quantization drift in saved images.

Bad (do not use for recon-FID):

```bash
python ./scripts/build_vae_recon_dataset.py ... --deterministic --half
```

Good:

```bash
python ./scripts/build_vae_recon_dataset.py ... --deterministic
```

## Optional: use a direct checkpoint path

If the stable pointer is broken on your environment, point to the run-local checkpoint directly:

- `./runs/VAE/vae_full128_reconfid_v2_<timestamp>/model_best.pt`
