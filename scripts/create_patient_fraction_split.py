import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def extract_patient_id(path: Path) -> str:
    name = path.name
    parts = name.split("-")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    parent = path.parent.name
    return parent if parent else path.stem


def extract_class(path: Path) -> str:
    token = path.name.split("-")[0].upper()
    if token:
        return token
    return path.parent.name.upper()


def write_lines(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic patient subset + split lists")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    all_images = list_images(args.input_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    patient_to_images: dict[str, list[Path]] = {}
    for p in all_images:
        pid = extract_patient_id(p)
        patient_to_images.setdefault(pid, []).append(p)

    all_patients = sorted(patient_to_images.keys())
    subset_rng = random.Random(args.seed)
    subset_rng.shuffle(all_patients)

    keep_count = max(1, int(round(len(all_patients) * args.fraction)))
    subset_patients = sorted(all_patients[:keep_count])

    # Mirror train_vae split logic: random.sample over sorted patient list
    num_train = int(len(subset_patients) * args.train_ratio)
    split_rng = random.Random(args.seed)
    train_patients = sorted(split_rng.sample(list(subset_patients), num_train))
    train_set = set(train_patients)
    val_patients = sorted([pid for pid in subset_patients if pid not in train_set])

    subset_images = sorted([str(p) for pid in subset_patients for p in patient_to_images[pid]])
    train_images = sorted([str(p) for pid in train_patients for p in patient_to_images[pid]])
    val_images = sorted([str(p) for pid in val_patients for p in patient_to_images[pid]])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_lines(out_dir / "subset_patients.txt", subset_patients)
    write_lines(out_dir / "train_patients.txt", train_patients)
    write_lines(out_dir / "val_patients.txt", val_patients)
    write_lines(out_dir / "subset_images.txt", subset_images)
    write_lines(out_dir / "train_images.txt", train_images)
    write_lines(out_dir / "val_images.txt", val_images)

    subset_class_counts = Counter(extract_class(Path(p)) for p in subset_images)
    train_class_counts = Counter(extract_class(Path(p)) for p in train_images)
    val_class_counts = Counter(extract_class(Path(p)) for p in val_images)

    stats = {
        "input_dir": str(args.input_dir),
        "fraction": float(args.fraction),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "num_total_patients": len(all_patients),
        "num_subset_patients": len(subset_patients),
        "num_train_patients": len(train_patients),
        "num_val_patients": len(val_patients),
        "num_total_images": len(all_images),
        "num_subset_images": len(subset_images),
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
        "subset_class_counts": dict(sorted(subset_class_counts.items())),
        "train_class_counts": dict(sorted(train_class_counts.items())),
        "val_class_counts": dict(sorted(val_class_counts.items())),
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
