# OCT-ready 4-class classifier — `train_classifier.py`

Train and evaluate an OCT-friendly image classifier for **4 classes (0–3)** with patient-wise stratified splits, optional mixing of **synthetic + real** data, balanced subsetting (including **true 0%**), and **auto-resolved input size/normalization/interpolation** for `timm` models (e.g., DINOv2 → 518×518).

---

## What this script does

- **Discovers images** in class folders `0/1/2/3` and extracts a **patient_id** from the token after the first underscore in the filename (e.g., `abc_123_xyz.jpg → 123`).
- **Splits by patient** (per class) into train/val to prevent leakage.
- **Optionally mixes** real and synthetic training images via `--real-mix`.
- **Optionally subsets** real/synthetic data per class by count or percent (supports **0%**).
- **Samples test data** (per class) from an independent test root.
- **Builds/loads models** from torchvision or `timm` (with auto data config).
- **Trains with AdamW + cosine LR**, class-weighted CE, and **OCT-friendly augmentations**.
- **Evaluates** on real val and sampled test, logging metrics per epoch.

---

## Expected data layout

```
<DATA_ROOT>/
├─ 0/  *.jpg|*.jpeg|*.png|*.bmp|*.tif|*.tiff|*.webp
├─ 1/  ...
├─ 2/  ...
└─ 3/  ...
```

**Patient ID rule**: token **after first underscore** → digits extracted. If none, token itself; else `"UNKNOWN"`.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision pillow numpy tqdm
# Optional (recommended for ViTs / DINOv2):
pip install timm
```

---

## Quick start

```bash
python train_classifier.py   --data-dir /path/to/REAL   --test-dir /path/to/TEST   --run-dir runs/exp1   --arch resnet18 --pretrained   --epochs 10 --batch-size 16 --lr 3e-4
```

### Using a timm model (auto size/normalization)
```bash
python train_classifier.py   --data-dir /path/to/REAL   --test-dir /path/to/TEST   --run-dir runs/dinov2   --arch vit_small_patch14_dinov2.lvd142m --pretrained
# Input will auto-resolve to 518 if required by the model.
```

---

## Sample commands

**Mix real & synthetic (70% real), subset synthetic to 25%:**
```bash
python train_classifier.py   --data-dir /path/to/REAL   --syn-data-dir /path/to/SYN   --test-dir /path/to/TEST   --run-dir runs/mix   --real-mix 0.7   --syn-subset-percent 25   --arch resnet50 --pretrained
```

**Three seeds (123, 124, 125):**
```bash
python train_classifier.py   --data-dir /path/to/REAL   --test-dir /path/to/TEST   --run-dir runs/kseeds   --k-seeds 3 --base-seed 123
```

**Use only 10% of REAL and skip test selection (0%):**
```bash
python train_classifier.py   --data-dir /path/to/REAL   --test-dir /path/to/TEST   --run-dir runs/subset   --subset-percent 10   --test-dir-amount 0.0
```

**Custom class weights:**
```bash
python train_classifier.py   --data-dir /path/to/REAL   --test-dir /path/to/TEST   --run-dir runs/weights   --class-weights "0.1,0.2,0.3,0.4"
```

---

## Key arguments (short list)

| Argument | Default | Purpose |
|---|---:|---|
| `--data-dir` | — | REAL data root (`0..3` subfolders). |
| `--test-dir` | — | Independent test root (`0..3`). |
| `--syn-data-dir` | `None` | Synthetic data root (`0..3`). |
| `--run-dir` | — | Where `.txt` logs are written. |
| `--arch` | `resnet18` | Torchvision or any `timm` model name. |
| `--pretrained` | `False` | Use pretrained backbone weights. |
| `--img-size` | `224` | Base size; **overridden** by `timm` cfg. |
| `--epochs` | `10` | Training epochs. |
| `--batch-size` | `8` | Batch size. |
| `--lr` | `3e-4` | Learning rate (AdamW). |
| `--weight-decay` | `0.05` | Weight decay (AdamW). |
| `--val-ratio` | `0.2` | Fraction of **patients per class** for validation. |
| `--test-dir-amount` | `0.10` | Fraction per class sampled from test (0.0–1.0). |
| `--real-mix` | `1.0` | Share of REAL in training mix (0.0–1.0). |
| `--subset-n-per-class` | `None` | Cap REAL per-class count (pre-split). |
| `--subset-percent` | `None` | Use P% REAL per class (pre-split, **0% ok**). |
| `--syn-subset-n-per-class` | `None` | Cap SYN per-class count (pre-mix). |
| `--syn-subset-percent` | `None` | Use P% SYN per class (pre-mix, **0% ok**). |
| `--k-seeds` | `3` | Number of independent runs. |
| `--base-seed` | `42` | First seed; others are `base_seed+i`. |
| `--device` | auto | `cuda` if available, else `cpu`. |
| `--class-weights` | `0.25,0.25,0.25,0.25` | CE weights for classes `0..3`. |

> Tip: `python train_classifier.py -h` shows the full help.

---

## Code map (short descriptions)

- **`set_global_seed(seed, deterministic=True)`** – Seed Python/NumPy/PyTorch/CUDA.
- **`timestamp()`** – Compact time string for filenames.
- **`parse_patient_id(filename)`** – Get patient ID from token after first underscore.
- **`discover_images(root, classes)`** – List `(filepath, label, patient_id)` from class dirs.
- **`apply_class_subset(items, subset_n_per_class, subset_percent, rng)`** – Balanced pre-split subsetting (REAL/SYN), supports 0%.
- **`stratified_patient_split(items, val_ratio, rng)`** – Patient-wise per-class split into train/val.
- **`sample_fraction_per_class(items, frac, rng)`** – Per-class random sampling for test subset (supports 0.0).
- **`ImageListDataset(items, transform)`** – Loads images, converts grayscale → RGB, applies transforms.
- **`create_model(arch, num_classes, pretrained)`** – Builds torchvision or `timm` model; sets classifier head to 4 classes.
- **`evaluate(model, loader, device)`** – Avg loss/acc on a dataloader (no-grad).
- **`run_one_training(...)`** – Full training routine: resolves `timm` data cfg (size/mean/std/interp), defines OCT augs, mixes data, trains with AdamW + cosine LR + class-weighted CE, tracks best val, evaluates test, writes logs.
- **`main()`** – CLI: parse args → discover/subset/split/sample → run seeds → write logs.

---

## Transforms (OCT-friendly, summarized)

- **Train**: gentle crops (0.85–1.0), small flips/rotations/affine, blur, slight sharpness/autocontrast, normalize, light random erasing.
- **Eval/Test**: resize to **effective** size (from `timm` if provided), then normalize.
- Grayscale images are **converted to RGB**.

---

## Outputs

For each seed, a log like:
```
runs/<name>/<timestamp>_<arch>_seed<seed>.txt
```
Contains setup (arch, effective input size, mean/std), dataset counts, per-epoch `train/val` metrics, best val epoch, and final test metrics.

---

## License

Use and adapt for research/experiments. Please cite appropriately if you publish results with this script.
