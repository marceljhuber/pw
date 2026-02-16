# VAE Implementation Gap Analysis and Action Plan

This report summarizes differences between the current OCT VAE pipeline and the original MAISI intent, why quality is currently poor, and what to run next.

## Scope

- Code reviewed:
  - `train_vae.py`
  - `scripts/utils_data.py`
  - `scripts/encode_to_latents.py`
  - `networks/autoencoderkl_maisi.py`
  - current 128px run config in `maisi_runs/.../config_VAE_fraction_consistent.json`
- Metrics reviewed:
  - train/test FID runs
  - VAE reconstruction sanity check

## What Differs vs Original MAISI Intent

## 1) Data regime and objective mismatch

- Original MAISI is foundation-style, large-scale, 3D CT pretraining.
- This project is 2D OCT adaptation with smaller effective data and stronger domain shift.
- Practical impact: weaker latent prior + weaker generalization in low-data regime.

## 2) Aggressive VAE train augmentation in current OCT pipeline

- Current train transform can include crop, rotation, color jitter, and speckle noise.
- For reconstruction-first latent learning, this can reduce faithful pixel-structure recovery.
- This is often acceptable for classification but can hurt latent diffusion quality.

## 3) Architecture capacity is moderate for fidelity target

- Current VAE uses `num_channels=[64,128,256]`, no broad attention.
- This is efficient, but may underfit fine OCT texture and layer boundaries for low FID targets.

## 4) GAN pressure may be too strong too early

- `adv_weight` was relatively high in baseline configs.
- Without explicit adversarial warmup scheduling, reconstruction can destabilize.

## 5) Evaluation feature mismatch

- FID currently uses default ImageNet Inception features (`torch-fidelity` default).
- OCT morphology is not ideal for that embedding; absolute FID values are inflated.

## Why Quality Is Poor Right Now

Main causes ranked by confidence:

1. VAE reconstruction quality ceiling (already high recon-FID in sanity checks).
2. Diffusion trained on latents produced by that VAE ceiling.
3. Domain shift between training subset and test distribution.
4. Unconditional generation evaluated per-class.
5. ImageNet feature extractor mismatch for OCT.

## New 128px Full-Data VAE Configs Added

All new configs use 100% of train data at 128x128.

1. `configs/config_VAE_full128_fidelity.json`
   - Goal: maximize reconstruction fidelity baseline.
   - No augmentation/noise.
   - Lower GAN weight (`adv_weight=0.02`).

2. `configs/config_VAE_full128_balanced.json`
   - Goal: compromise between fidelity and robustness.
   - Mild augmentation/noise.
   - Moderate GAN weight (`adv_weight=0.03`).

3. `configs/config_VAE_full128_capacity.json`
   - Goal: higher-quality decoder behavior with conservative adversarial pressure.
   - No augmentation/noise.
   - Attention on deepest stage, checkpointing on, lower LR, longer training.
   - Lower GAN weight (`adv_weight=0.01`).

## Supporting Code Change

- `scripts/utils_data.py` now supports disabling random crop explicitly with:
  - `"random_crop_scale": null`
- This was needed to produce a true no-augmentation fidelity config.

## Recommended Run Order

1. Train `config_VAE_full128_fidelity.json` first (fastest clean baseline).
2. Run VAE recon sanity FID (real vs recon on train-balanced subset).
3. If improved, re-encode latents and retrain diffusion with checkpoint FID monitor.
4. Then run `config_VAE_full128_capacity.json` if more quality is needed.
5. Keep `balanced` config as fallback for robustness/generalization.

## Success Criteria

- VAE recon sanity FID should drop materially vs current baseline.
- Diffusion checkpoint monitor should show better best-FID than prior run.
- Prefer early-stop at best monitored checkpoint rather than final epoch.
