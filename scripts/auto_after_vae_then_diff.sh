#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ROOT>"
  exit 1
fi

RUN_ROOT="$1"
VAE_CFG="$RUN_ROOT/configs/config_VAE_fraction_consistent.json"
DIFF_CFG="$RUN_ROOT/configs/config_DIFF_fraction_consistent.json"
TRAIN_DIR="/media/user/Extreme SSD/Thesis/data/KermanyV3_resized/train"
LAT_TRAIN="$RUN_ROOT/latents/train"
LAT_VAL="$RUN_ROOT/latents/val"
SPLIT_TRAIN="$RUN_ROOT/splits/train_patients.txt"
SPLIT_VAL="$RUN_ROOT/splits/val_patients.txt"
REAL_TEST_ROOT="/media/user/Extreme SSD/Thesis/data/KermanyV3_resized/test"

ENABLE_FID_MONITOR="${ENABLE_FID_MONITOR:-true}"
FID_EVAL_EVERY="${FID_EVAL_EVERY:-100}"
FID_SAMPLES_PER_CLASS="${FID_SAMPLES_PER_CLASS:-250}"
FID_STEPS="${FID_STEPS:-250}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
FID_POLL_SECONDS="${FID_POLL_SECONDS:-180}"

mkdir -p "$RUN_ROOT/logs" "$LAT_TRAIN" "$LAT_VAL" "$RUN_ROOT/DIFFUSION"

echo "Waiting for VAE process to finish..."
while pgrep -f "train_vae.py --config $VAE_CFG" >/dev/null; do
  sleep 120
done

echo "Selecting best available VAE checkpoint..."
AE_CKPT="$(python3 - <<'PY' "$RUN_ROOT"
from pathlib import Path
import re
import sys

run_root = Path(sys.argv[1])
vae_root = run_root / "VAE"
dirs = sorted([p for p in vae_root.iterdir() if p.is_dir()])
if not dirs:
    raise SystemExit("No VAE run directory found")
latest_dir = dirs[-1]
best = latest_dir / "model_best.pt"
if best.exists():
    print(str(best))
else:
    cands = sorted(latest_dir.glob("vae_fraction_consistent_*.pt"))
    bestp = None
    bestn = -1
    for p in cands:
        m = re.search(r"_(\d+)\.pt$", p.name)
        if m and int(m.group(1)) > bestn:
            bestn = int(m.group(1))
            bestp = p
    if bestp is None:
        raise SystemExit("No epoch checkpoint found")
    print(str(bestp))
PY
)"

echo "Using VAE checkpoint: $AE_CKPT"

python3 - <<'PY' "$DIFF_CFG" "$AE_CKPT"
import json
import sys

cfg_path, ae_ckpt = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
cfg.setdefault("main", {})["trained_autoencoder_path"] = ae_ckpt
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("Updated trained_autoencoder_path ->", ae_ckpt)
PY

echo "Encoding TRAIN latents..."
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/encode_to_latents.py" \
  --input_dir "$TRAIN_DIR" \
  --output_dir "$LAT_TRAIN" \
  --autoencoder_path "$AE_CKPT" \
  --vae_config "$VAE_CFG" \
  --image_size 128 \
  --patient_list_path "$SPLIT_TRAIN" \
  --seed 42

echo "Encoding VAL latents..."
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/encode_to_latents.py" \
  --input_dir "$TRAIN_DIR" \
  --output_dir "$LAT_VAL" \
  --autoencoder_path "$AE_CKPT" \
  --vae_config "$VAE_CFG" \
  --image_size 128 \
  --patient_list_path "$SPLIT_VAL" \
  --seed 42

echo "Starting diffusion training..."
WANDB_MODE=online WANDB_SILENT=false WANDB_PROJECT=maisi-full-training \
  conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/train_diffusion.py" \
  --config "$DIFF_CFG" \
  --name "diff_fraction_consistent" \
  --run_dir "$RUN_ROOT/DIFFUSION" \
  > "$RUN_ROOT/logs/diffusion_auto.log" 2>&1 &
DIFF_PID=$!

if [[ "$ENABLE_FID_MONITOR" == "true" ]]; then
  echo "Starting checkpoint FID monitor (space-saving mode)..."
  conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/monitor_diffusion_fid.py" \
    --run_root "$RUN_ROOT" \
    --diff_runs_root "$RUN_ROOT/DIFFUSION" \
    --diff_name "diff_fraction_consistent" \
    --diff_config "$DIFF_CFG" \
    --real_root "$REAL_TEST_ROOT" \
    --eval_every_epochs "$FID_EVAL_EVERY" \
    --samples_per_class "$FID_SAMPLES_PER_CLASS" \
    --steps "$FID_STEPS" \
    --batch_size "$FID_BATCH_SIZE" \
    --poll_seconds "$FID_POLL_SECONDS" \
    --stop_when_idle \
    > "$RUN_ROOT/logs/diff_fid_monitor.log" 2>&1 &
  MONITOR_PID=$!
fi

wait "$DIFF_PID"

if [[ "$ENABLE_FID_MONITOR" == "true" ]]; then
  wait "$MONITOR_PID"
fi
