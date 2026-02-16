# FID Learnings and Fix Plan (OCT)

This document captures practical findings from recent debugging runs and what to do next to reduce FID reliably.

## Executive Summary

- High FID is **not** from one single bug anymore; it is a combination of:
  - VAE reconstruction quality ceiling,
  - diffusion sample quality,
  - OCT vs ImageNet feature mismatch in `torch-fidelity`,
  - and evaluation protocol choices (classwise on unconditional models).
- Biggest correctness bug found and fixed: latent-size mismatch at generation time (now auto-inferred and validated).

## Key Findings From Recent Runs

## 1) Latent shape mismatch was real and damaging

- Problem: generation script previously defaulted to fixed latent size and could silently mismatch training latent shape.
- Fix: `scripts/generate_diffusion_dataset.py` now infers latent H/W from `main.latents_path` and fails on mismatch.

## 2) Unconditional model + per-class FID is harsh/misaligned

- We generate unconditionally, then split outputs into class folders for evaluation.
- Per-class FID becomes high because there is no explicit class control.
- This is expected behavior; pooled (`ALL`) FID is often lower.

## 3) Domain shift matters a lot

- Against test split (real test vs fake): `ALL FID` around ~88-90 for the tested checkpoint.
- Against train-25% matched real subset: `ALL FID` dropped to ~64.
- Conclusion: train-to-test shift contributes significantly.

## 4) VAE is already a major bottleneck

- Sanity check with real-vs-VAE-recon (train-25% matched 250/class) gave:
  - `ALL FID = 65.7402`
- This means diffusion is not the only source of error; VAE reconstruction quality is already limiting downstream FID.

## 5) FID monitor found non-monotonic diffusion behavior

- During checkpoint sweep, FID improved up to around epoch 900, then worsened at epoch 1000.
- Best monitored checkpoint in that run: epoch 900 (`diff_899.pt`).

## 6) Normalization path is now consistent

- Train/val transforms use `ToTensor()` + `Lambda(2x-1)` => `[-1, 1]`.
- Latent encoding uses deterministic val-style preprocessing (no random aug/noise).
- Sampling/PNG conversion now consistently maps `[-1, 1] -> [0, 255]` only at save/export points.

## What To Change To Reduce FID

Prioritized (highest impact first):

1. **Improve Stage-1 VAE quality first**
   - Track and select VAE by reconstruction metrics and recon-FID sanity checks.
   - Avoid using partially trained or unstable VAE checkpoints for latent encoding.

2. **Select diffusion checkpoints by monitored FID, not last epoch**
   - Keep checkpoint sweep monitoring (every N epochs).
   - Use best checkpoint (e.g., epoch 900), not automatically epoch 1000.

3. **Use fair evaluation protocol**
   - Compare with matched counts where possible.
   - For unconditional models, treat per-class FID carefully and prioritize pooled metrics.

4. **Move to OCT-domain feature extractor for FID (recommended)**
   - Default Inception (ImageNet) is suboptimal for OCT semantics.
   - Register/use OCT feature extractor with `compute_oct_metrics.py --feature_extractor ...`.

5. **Optional: conditional+CFG only when fully wired and needed**
   - Conditional setup is slower and more fragile.
   - Use unconditional baseline first, then add conditional+CFG with proper inference guidance.

## Current Automation in Repo

- `scripts/auto_after_vae_then_diff.sh`
  - waits for VAE end,
  - encodes train/val latents,
  - starts diffusion,
  - starts checkpoint FID monitor.

- `scripts/monitor_diffusion_fid.py`
  - evaluates every N epochs,
  - logs to CSV and best-state JSON,
  - deletes temporary generated eval images after each eval to save disk.

## Suggested Next Iteration

1. Freeze current best diffusion checkpoint from monitor.
2. Run a dedicated VAE checkpoint sweep for recon-FID sanity metric.
3. Re-encode latents with best VAE only.
4. Retrain diffusion with same monitoring and stop early at best FID plateau.
5. Recompute final metrics with both:
   - standard `torch-fidelity` (for comparability), and
   - OCT-domain extractor (for meaningful absolute quality).
