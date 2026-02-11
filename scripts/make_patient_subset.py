import argparse
import glob
import os
import random
from pathlib import Path


def list_image_files(directory_path: str):
    exts = (".jpg", ".jpeg", ".png")
    files = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)
    return [f for f in files if f.lower().endswith(exts)]


def extract_patient_id(path_str: str) -> str:
    name = Path(path_str).name
    parts = name.split("-")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return Path(path_str).stem


def extract_class_name(path_str: str) -> str:
    # Prefer filename prefix (e.g. CNV-..., DME-...), else parent dir name.
    name = Path(path_str).name
    prefix = name.split("-")[0]
    if prefix:
        return prefix
    return Path(path_str).parent.name


def main():
    ap = argparse.ArgumentParser(description="Create deterministic patient subset list")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stratify_by_class",
        action="store_true",
        help="Sample fraction per class prefix (recommended).",
    )
    args = ap.parse_args()

    if not (0 < args.fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    files = list_image_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No images found under {args.input_dir}")

    rng = random.Random(args.seed)

    if args.stratify_by_class:
        class_to_pids = {}
        for f in files:
            cls = extract_class_name(f)
            pid = extract_patient_id(f)
            class_to_pids.setdefault(cls, set()).add(pid)

        keep = set()
        for cls, pids in sorted(class_to_pids.items()):
            pids = sorted(pids)
            rng.shuffle(pids)
            k = max(1, int(round(len(pids) * args.fraction)))
            keep.update(pids[:k])
    else:
        pids = sorted({extract_patient_id(f) for f in files})
        rng.shuffle(pids)
        k = max(1, int(round(len(pids) * args.fraction)))
        keep = set(pids[:k])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(keep)) + "\n")
    print(f"Wrote {len(keep)} patient ids to {out}")


if __name__ == "__main__":
    main()
