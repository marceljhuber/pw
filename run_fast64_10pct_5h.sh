#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/media/user/Extreme SSD/Thesis/maisi_runs/maisi_fast64_10pct_20260208_144118"

VAE_CONFIG="/media/user/Extreme SSD/Thesis/pw/configs/config_VAE_fast64_10pct_5h.json"
DIFF_CONFIG="/media/user/Extreme SSD/Thesis/pw/configs/config_DIFF_fast64_10pct_5h.json"
CTRL_CONFIG="/media/user/Extreme SSD/Thesis/pw/configs/config_CONTROLNET_fast64_10pct_5h.json"

DATA_DIR="/media/user/Extreme SSD/Thesis/data/KermanyV3_resized/train"
LATENTS_DIR="$RUN_ROOT/latents/train"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$LATENTS_DIR" "$RUN_ROOT/DIFFUSION" "$RUN_ROOT/CONTROLNET"

# Keep runs self-contained for later reproduction
mkdir -p "$RUN_ROOT/configs"
cp -f "$VAE_CONFIG" "$RUN_ROOT/configs/" || true
cp -f "$DIFF_CONFIG" "$RUN_ROOT/configs/" || true
cp -f "$CTRL_CONFIG" "$RUN_ROOT/configs/" || true

export WANDB_MODE=disabled
export WANDB_SILENT=true

echo "Run root: $RUN_ROOT"
echo "Latents: $LATENTS_DIR"
date

echo "[1/4] Train VAE"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/train_vae.py" --config "$VAE_CONFIG" \
  2>&1 | tee "$RUN_ROOT/logs/vae.log"

echo "[2/4] Encode images -> latents (subset patients=0.1)"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/encode_to_latents.py" \
  --input_dir "$DATA_DIR" \
  --output_dir "$LATENTS_DIR" \
  --autoencoder_path "$RUN_ROOT/VAE/vae_fast64_10pct_best.pt" \
  --vae_config "$VAE_CONFIG" \
  --image_size 64 \
  --subset_patient_fraction 0.1 \
  2>&1 | tee "$RUN_ROOT/logs/encode_latents.log"

echo "[3/4] Train diffusion UNet"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/train_diffusion.py" \
  --config "$DIFF_CONFIG" \
  --name "diffusion_fast64_10pct" \
  --run_dir "$RUN_ROOT/DIFFUSION" \
  2>&1 | tee "$RUN_ROOT/logs/diffusion.log"

echo "[4/4] Train ControlNet"
conda run -n maisi python -m scripts.train_controlnet \
  --config_path "$CTRL_CONFIG" \
  -g 1 \
  2>&1 | tee "$RUN_ROOT/logs/controlnet.log"

echo "DONE"
date
