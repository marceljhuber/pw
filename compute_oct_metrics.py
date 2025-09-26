"""
Compute per-class AND overall FID, IS, and SSIM in one go.

Folder layout (both roots):
    real_root/
        0/ 1/ 2/ 3/   # class folders (any image formats)
    fake_root/
        0/ 1/ 2/ 3/

Outputs:
- Neat console table with per-class + overall metrics
- A timestamped TXT report with all details (paths, date/time, seed, counts)

Requires:
    pip install torch-fidelity scikit-image pillow
"""

import argparse
import os
import sys
import math
import random
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

import torch
import torch_fidelity


# ------------------------------- Utilities ---------------------------------- #

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images_recursive(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def ensure_class_dirs(root: Path, classes=("0", "1", "2", "3")):
    ok = True
    missing = []
    for c in classes:
        if not (root / c).exists():
            ok = False
            missing.append(c)
    return ok, missing


def human(num):
    return f"{num:.4f}" if isinstance(num, (float, np.floating)) else str(num)


def load_gray_01(path: Path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def mean_ssim_dir(dir_real: Path, dir_fake: Path):
    """Pair images by sorted filename; if different counts, use the min length.
    Resize fake to real HxW (anti-aliased) to allow SSIM; compute mean."""
    r_files = list_images_recursive(dir_real)
    f_files = list_images_recursive(dir_fake)

    n = min(len(r_files), len(f_files))
    scores = []
    for r, f in zip(r_files[:n], f_files[:n]):
        x = load_gray_01(r)
        y = load_gray_01(f)
        if x.shape != y.shape:
            y = resize(
                y,
                x.shape,
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )
        scores.append(ssim(x, y, data_range=1.0))
    return (float(np.mean(scores)) if scores else float("nan")), len(scores)


def compute_fid_is(reals, fakes, device_cuda=True, feature_extractor=None):
    """
    Use torch-fidelity to compute FID (real vs fake) and IS (fake only).
    `reals` and `fakes` are directory paths.
    """
    kwargs = dict(
        input1=str(fakes),  # IS uses input1 (generated set)
        input2=str(reals),  # needed for FID
        cuda=device_cuda,
        fid=True,
        isc=True,
        kid=False,
        verbose=False,
        samples_find_deep=True,  # recurse into subfolders
    )
    if feature_extractor:
        # torch-fidelity will fall back to default if unknown; typical is 'inception-v3-compat'
        kwargs["feature_extractor"] = feature_extractor

    m = torch_fidelity.calculate_metrics(**kwargs)
    fid = float(m["frechet_inception_distance"])
    is_mean = float(m["inception_score_mean"])
    is_std = float(m["inception_score_std"])
    return fid, is_mean, is_std


def print_table(rows, title="Metrics"):
    if not rows:
        return
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    line = lambda r: " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{title}")
    print(line(rows[0]))
    print(sep)
    for r in rows[1:]:
        print(line(r))
    print()


# ------------------------------- Main logic --------------------------------- #


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-class + overall FID, IS, SSIM for OCT datasets."
    )
    ap.add_argument(
        "--real_root",
        required=True,
        type=Path,
        help="Path to REAL images root (contains 0..3)",
    )
    ap.add_argument(
        "--fake_root",
        required=True,
        type=Path,
        help="Path to FAKE images root (contains 0..3)",
    )
    ap.add_argument(
        "--feature_extractor",
        default=None,
        type=str,
        help="torch-fidelity feature extractor for FID (e.g., 'inception-v3-compat' or your OCT extractor)",
    )
    ap.add_argument(
        "--seed", default=42, type=int, help="Random seed (affects IS splits)"
    )
    ap.add_argument(
        "--tz", default="Europe/Vienna", help="Timezone for report timestamp"
    )
    ap.add_argument(
        "--out_dir",
        default=Path("./metrics_reports"),
        type=Path,
        help="Directory to save the TXT report",
    )
    ap.add_argument(
        "--no_cuda", action="store_true", help="Force CPU for torch-fidelity"
    )
    args = ap.parse_args()

    # Validate
    for root in [args.real_root, args.fake_root]:
        if not root.exists():
            print(f"ERROR: {root} does not exist.", file=sys.stderr)
            sys.exit(1)
        ok, missing = ensure_class_dirs(root)
        if not ok:
            print(f"WARNING: {root} missing class dirs: {missing}", file=sys.stderr)

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    classes = ["0", "1", "2", "3"]

    # Counts
    counts = {}
    for c in classes:
        r_dir = args.real_root / c
        f_dir = args.fake_root / c
        counts[c] = (
            len(list_images_recursive(r_dir)),
            len(list_images_recursive(f_dir)),
        )
    overall_counts = (
        sum(counts[c][0] for c in classes),
        sum(counts[c][1] for c in classes),
    )

    # Per-class metrics
    per_class = {}
    for c in classes:
        r_dir = args.real_root / c
        f_dir = args.fake_root / c
        fid, is_mean, is_std = compute_fid_is(
            r_dir, f_dir, device_cuda=use_cuda, feature_extractor=args.feature_extractor
        )
        ssim_mean, ssim_pairs = mean_ssim_dir(r_dir, f_dir)
        per_class[c] = dict(
            FID=fid,
            IS=is_mean,
            IS_STD=is_std,
            SSIM=ssim_mean,
            SSIM_pairs=ssim_pairs,
            n_real=counts[c][0],
            n_fake=counts[c][1],
        )

    # Overall metrics (compute on union of all images)
    fid_all, is_all_mean, is_all_std = compute_fid_is(
        args.real_root,
        args.fake_root,
        device_cuda=use_cuda,
        feature_extractor=args.feature_extractor,
    )
    ssim_all_mean, ssim_all_pairs = mean_ssim_dir(args.real_root, args.fake_root)

    # Pretty print
    header = ["Class", "n_real", "n_fake", "FID", "IS (±std)", "SSIM", "SSIM pairs"]
    rows = [header]
    for c in classes:
        pc = per_class[c]
        rows.append(
            [
                c,
                str(pc["n_real"]),
                str(pc["n_fake"]),
                human(pc["FID"]),
                f"{human(pc['IS'])} (±{human(pc['IS_STD'])})",
                human(pc["SSIM"]),
                str(pc["SSIM_pairs"]),
            ]
        )
    rows.append(
        [
            "ALL",
            str(overall_counts[0]),
            str(overall_counts[1]),
            human(fid_all),
            f"{human(is_all_mean)} (±{human(is_all_std)})",
            human(ssim_all_mean),
            str(ssim_all_pairs),
        ]
    )
    print_table(rows, title="Per-class and Overall Metrics")

    # Build report text
    tz = ZoneInfo(args.tz)
    now = datetime.now(tz)
    lines = []
    lines.append("OCT Metrics Report")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {now.isoformat(timespec='seconds')} ({args.tz})")
    lines.append(f"Seed: {args.seed}")
    lines.append(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    lines.append(
        f"Feature extractor (FID): {args.feature_extractor or 'inception-v3-compat (default)'}"
    )
    lines.append(f"Real root: {args.real_root.resolve()}")
    lines.append(f"Fake root: {args.fake_root.resolve()}")
    lines.append("-" * 80)
    lines.append("{:<8} {:>8} {:>8} {:>12} {:>18} {:>10} {:>12}".format(*header))
    for c in classes:
        pc = per_class[c]
        lines.append(
            "{:<8} {:>8} {:>8} {:>12.4f} {:>9.4f} (±{:>6.4f}) {:>10.4f} {:>12}".format(
                c,
                pc["n_real"],
                pc["n_fake"],
                pc["FID"],
                pc["IS"],
                pc["IS_STD"],
                pc["SSIM"],
                pc["SSIM_pairs"],
            )
        )
    lines.append("-" * 80)
    lines.append(
        "{:<8} {:>8} {:>8} {:>12.4f} {:>9.4f} (±{:>6.4f}) {:>10.4f} {:>12}".format(
            "ALL",
            overall_counts[0],
            overall_counts[1],
            fid_all,
            is_all_mean,
            is_all_std,
            ssim_all_mean,
            ssim_all_pairs,
        )
    )
    lines.append("=" * 80)
    lines.append("Notes:")
    lines.append(
        "- FID is computed on the union of all images for the 'ALL' row (do not average per-class FIDs)."
    )
    lines.append(
        "- IS is computed on generated images only; per-class IS uses the class subfolder, 'ALL' uses the union."
    )
    lines.append(
        "- SSIM pairs images by sorted filename; when counts differ, only min(count_real, count_fake) pairs are used."
    )
    lines.append(
        "- For OCT-domain FID, provide --feature_extractor pointing to your registered OCT feature extractor in torch-fidelity."
    )

    # Save report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"metrics_summary_{now.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}.txt"
    out_path = args.out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved report to: {out_path.resolve()}")


if __name__ == "__main__":
    """
    Example:
      python compute_oct_metrics.py \
        --real_root /data/oct/real \
        --fake_root /data/oct/fake \
        --feature_extractor inception-v3-compat \
        --seed 123 \
        --out_dir ./metrics_reports
    """
    main()
