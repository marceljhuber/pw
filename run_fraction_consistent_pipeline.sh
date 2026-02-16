#!/usr/bin/env bash
set -euo pipefail

# Consistent fractional pipeline:
# - Patient-level subset and split (deterministic)
# - VAE trains on the subset and uses its train/val split
# - Latent encoding reuses the exact same patient split:
#     * train patients -> latents/train (used for diffusion)
#     * val patients   -> latents/val   (encoded, but NOT used for diffusion)

DATA_TRAIN="${DATA_TRAIN:-/media/user/Extreme SSD/Thesis/data/KermanyV3_resized/train}"
DATA_TEST="${DATA_TEST:-/media/user/Extreme SSD/Thesis/data/KermanyV3_resized/test}"
FRACTION="${FRACTION:-0.25}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-42}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
VAE_BATCH="${VAE_BATCH:-64}"
DIFF_BATCH="${DIFF_BATCH:-64}"
DIFF_STEPS="${DIFF_STEPS:-1000}"
GEN_BATCH="${GEN_BATCH:-32}"
GEN_PER_CLASS="${GEN_PER_CLASS:-1000}"
VAE_EPOCHS="${VAE_EPOCHS:-100}"
DIFF_EPOCHS="${DIFF_EPOCHS:-1000}"
ENABLE_CONDITIONAL="${ENABLE_CONDITIONAL:-false}"
USE_CFG="${USE_CFG:-false}"
CFG_DROPOUT_PROB="${CFG_DROPOUT_PROB:-0.15}"

RUN_ROOT="${RUN_ROOT:-/media/user/Extreme SSD/Thesis/maisi_runs/frac_consistent_$(date +%Y%m%d_%H%M%S)}"
SPLIT_DIR="$RUN_ROOT/splits"
LATENTS_TRAIN_DIR="$RUN_ROOT/latents/train"
LATENTS_VAL_DIR="$RUN_ROOT/latents/val"

VAE_BASE_CFG="${VAE_BASE_CFG:-/media/user/Extreme SSD/Thesis/pw/configs/config_VAE.json}"
DIFF_BASE_CFG="${DIFF_BASE_CFG:-/media/user/Extreme SSD/Thesis/pw/configs/config_DIFF.json}"
VAE_CFG="$RUN_ROOT/configs/config_VAE_fraction_consistent.json"
DIFF_CFG="$RUN_ROOT/configs/config_DIFF_fraction_consistent.json"

JOBNAME="${JOBNAME:-vae_fraction_consistent}"
DIFF_NAME="${DIFF_NAME:-diff_fraction_consistent}"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/configs" "$RUN_ROOT/logs" "$RUN_ROOT/DIFFUSION" "$LATENTS_TRAIN_DIR" "$LATENTS_VAL_DIR"

echo "Run root: $RUN_ROOT"
echo "[1/8] Build deterministic patient split lists"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/create_patient_fraction_split.py" \
  --input_dir "$DATA_TRAIN" \
  --output_dir "$SPLIT_DIR" \
  --fraction "$FRACTION" \
  --train_ratio "$TRAIN_RATIO" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_ROOT/logs/split.log"

echo "[2/8] Materialize run configs"
python3 - <<'PY' "$VAE_BASE_CFG" "$DIFF_BASE_CFG" "$VAE_CFG" "$DIFF_CFG" "$RUN_ROOT" "$DATA_TRAIN" "$LATENTS_TRAIN_DIR" "$JOBNAME" "$VAE_BATCH" "$DIFF_BATCH" "$IMAGE_SIZE" "$SEED" "$SPLIT_DIR" "$VAE_EPOCHS" "$DIFF_EPOCHS" "$ENABLE_CONDITIONAL" "$USE_CFG" "$CFG_DROPOUT_PROB"
import json
import sys

(
    vae_base,
    diff_base,
    vae_out,
    diff_out,
    run_root,
    data_train,
    latents_train,
    jobname,
    vae_batch,
    diff_batch,
    image_size,
    seed,
    split_dir,
    vae_epochs,
    diff_epochs,
    enable_conditional,
    use_cfg,
    cfg_dropout_prob,
) = sys.argv[1:]


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

with open(vae_base) as f:
    vae = json.load(f)
vae.setdefault("main", {})["run_dir"] = run_root
vae["main"]["jobname"] = jobname
vae.setdefault("training", {})["batch_size"] = int(vae_batch)
vae["training"]["epochs"] = int(vae_epochs)
vae["training"]["seed"] = int(seed)
vae.setdefault("data", {})["image_dir"] = data_train
vae["data"]["patient_list_path"] = f"{split_dir}/subset_patients.txt"
vae["data"].pop("subset_patient_fraction", None)
vae.setdefault("data", {}).setdefault("train_transform", {})["resize"] = [int(image_size), int(image_size)]
vae.setdefault("data", {}).setdefault("val_transform", {})["resize"] = [int(image_size), int(image_size)]

with open(vae_out, "w") as f:
    json.dump(vae, f, indent=2)

with open(diff_base) as f:
    diff = json.load(f)
diff.setdefault("main", {})["image_dir"] = data_train
diff["main"]["latents_path"] = latents_train
diff["main"]["trained_autoencoder_path"] = f"{run_root}/VAE/{jobname}_best.pt"
diff["main"]["enable_conditional_training"] = parse_bool(enable_conditional)
diff["main"]["use_cfg"] = parse_bool(use_cfg)
diff["main"]["cfg_dropout_prob"] = float(cfg_dropout_prob)
diff.setdefault("model_config", {}).setdefault("diffusion_unet_train", {})["batch_size"] = int(diff_batch)
diff["model_config"]["diffusion_unet_train"]["n_epochs"] = int(diff_epochs)
diff.setdefault("env_config", {})["model_dir"] = f"{run_root}/DIFFUSION/models"
diff["env_config"]["model_filename"] = "diff_unet_ckpt.pt"

with open(diff_out, "w") as f:
    json.dump(diff, f, indent=2)
PY

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-maisi-fraction-consistent}"

echo "[3/8] Train VAE on subset patients"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/train_vae.py" \
  --config "$VAE_CFG" \
  2>&1 | tee "$RUN_ROOT/logs/vae.log"

echo "[4/8] Encode TRAIN split latents (for diffusion training only)"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/encode_to_latents.py" \
  --input_dir "$DATA_TRAIN" \
  --output_dir "$LATENTS_TRAIN_DIR" \
  --autoencoder_path "$RUN_ROOT/VAE/${JOBNAME}_best.pt" \
  --vae_config "$VAE_CFG" \
  --image_size "$IMAGE_SIZE" \
  --patient_list_path "$SPLIT_DIR/train_patients.txt" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_ROOT/logs/encode_train_latents.log"

echo "[5/8] Encode VAL split latents (kept separate; not used by diffusion training)"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/encode_to_latents.py" \
  --input_dir "$DATA_TRAIN" \
  --output_dir "$LATENTS_VAL_DIR" \
  --autoencoder_path "$RUN_ROOT/VAE/${JOBNAME}_best.pt" \
  --vae_config "$VAE_CFG" \
  --image_size "$IMAGE_SIZE" \
  --patient_list_path "$SPLIT_DIR/val_patients.txt" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_ROOT/logs/encode_val_latents.log"

echo "[6/8] Train diffusion on TRAIN latents only"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/train_diffusion.py" \
  --config "$DIFF_CFG" \
  --name "$DIFF_NAME" \
  --run_dir "$RUN_ROOT/DIFFUSION" \
  2>&1 | tee "$RUN_ROOT/logs/diffusion.log"

echo "[7/8] Generate synthetic dataset (1000 per class by default)"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/scripts/generate_diffusion_dataset.py" \
  --out_root "$RUN_ROOT/generated" \
  --dataset_name "diffusion_only_1kpc" \
  --vae_ckpt "$RUN_ROOT/VAE/${JOBNAME}_best.pt" \
  --diff_ckpt "$RUN_ROOT/DIFFUSION/${DIFF_NAME}_best.pt" \
  --diff_config "$DIFF_CFG" \
  --steps "$DIFF_STEPS" \
  --batch_size "$GEN_BATCH" \
  --fixed_per_class "$GEN_PER_CLASS" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_ROOT/logs/generate_1k_per_class.log"

python3 - <<'PY' "$RUN_ROOT"
from pathlib import Path
run_root = Path(__import__('sys').argv[1])
src = run_root / "generated" / "diffusion_only_1kpc"
dst = run_root / "generated" / "diffusion_only_1kpc_eval_0to3"
dst.mkdir(parents=True, exist_ok=True)
mapping = {"CNV": "0", "DME": "1", "DRUSEN": "2", "NORMAL": "3"}
for k, v in mapping.items():
    s = src / k
    d = dst / v
    if d.exists() or d.is_symlink():
        d.unlink()
    d.symlink_to(s)
print(dst)
PY

echo "[8/8] Compute FID/IS/SSIM"
conda run -n maisi python "/media/user/Extreme SSD/Thesis/pw/compute_oct_metrics.py" \
  --real_root "$DATA_TEST" \
  --fake_root "$RUN_ROOT/generated/diffusion_only_1kpc_eval_0to3" \
  --out_dir "$RUN_ROOT/metrics_reports" \
  2>&1 | tee "$RUN_ROOT/logs/fid_metrics.log"

echo "DONE: $RUN_ROOT"
