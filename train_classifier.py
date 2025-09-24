#!/usr/bin/env python3
"""
train_classifier.py  —  OCT-ready 4-class classifier

Changes in this version:
- ### NEW/CHANGED ### Auto-resolve input size/mean/std/interpolation from timm model cfg (e.g., DINOv2 -> 518).
- ### NEW/CHANGED ### Subset & test sampling allow true 0% now.
- OCT-friendly augs maintained; grayscale handled via convert('RGB').
"""

import argparse
import time
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode  # ### NEW/CHANGED ###

# Optional timm support (e.g., DINOv2 ViTs)
try:
    import timm
    from timm.data import (
        resolve_model_data_config as timm_resolve_data_cfg,
    )  # ### NEW/CHANGED ###

    HAS_TIMM = True
except Exception:
    timm_resolve_data_cfg = None
    HAS_TIMM = False


# ---------------------------
# Utilities
# ---------------------------


def set_global_seed(seed: int, deterministic: bool = True):
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Make dataloader worker seeding reproducible
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp():
    import time

    return time.strftime("%Y%m%d-%H%M%S")


def parse_patient_id(filename: str) -> str:
    """
    Extract patient_id as the number after the first underscore.
    Example: "abc_123_xyz.jpg" -> "123"
    If not found, returns token after first underscore.
    """
    stem = Path(filename).stem
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return "UNKNOWN"
    token = parts[1]
    num = "".join(ch for ch in token if ch.isdigit())
    return num if num else token


def discover_images(root: Path, classes: List[str]) -> List[Tuple[str, int, str]]:
    """
    Return list of (filepath, label, patient_id)
    For directories that don’t encode patients (e.g., synthetic, test), patient_id may be "UNKNOWN".
    """
    items = []
    for cls in classes:
        class_dir = root / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory: {class_dir}")
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            }:
                pid = parse_patient_id(p.name)
                items.append((str(p), int(cls), pid))
    return items


def apply_class_subset(
    items: List[Tuple[str, int, str]],
    subset_n_per_class: Optional[int],
    subset_percent: Optional[float],
    rng: random.Random,
) -> List[Tuple[str, int, str]]:
    """
    Apply balanced subsetting BEFORE any splitting/sampling.
    ### NEW/CHANGED ### allows true 0% (yields 0).
    """
    if subset_n_per_class is not None and subset_percent is not None:
        raise ValueError(
            "Use either --subset-n-per-class OR --subset-percent (not both)."
        )

    if subset_n_per_class is None and subset_percent is None:
        return items

    by_class: Dict[int, List[Tuple[str, int, str]]] = {}
    for rec in items:
        by_class.setdefault(rec[1], []).append(rec)

    selected = []
    for cls, recs in by_class.items():
        pool = recs[:]
        rng.shuffle(pool)
        if subset_n_per_class is not None:
            k = max(0, min(subset_n_per_class, len(pool)))
        else:
            pct = float(subset_percent)
            if pct <= 0:
                k = 0
            elif pct >= 100:
                k = len(pool)
            else:
                k = int(round(len(pool) * (pct / 100.0)))
                if k == 0 and len(pool) > 0:
                    k = 1  # for very small fractions but >0
        selected.extend(pool[:k])

    return selected


from typing import Any, List, Sequence, Tuple, Dict, DefaultDict
from collections import defaultdict
import random


def stratified_patient_split(
    items: Sequence[Any], val_ratio: float, rng: random.Random
) -> Tuple[List[Any], List[Any]]:
    """
    Split items into train/val by patient IDs, stratified within each class.
    Items are expected to be strings like: "<class>-<patientid>-<...>".

    If class is unknown/missing (e.g., the string has no '-' or starts with '-'),
    the item is grouped under the '__ALL__' class.

    Args:
        items: A sequence of item identifiers (typically strings).
        val_ratio: Fraction of items to allocate to the validation split (0..1).
        rng: A random.Random instance controlling stochasticity.

    Returns:
        train_items, val_items: lists containing the original `items`.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

    # class -> patient -> list of items for that patient
    class_to_patient_items: Dict[str, DefaultDict[str, List[Any]]] = {}
    class_to_patient_items = defaultdict(lambda: defaultdict(list))

    def parse(item: Any) -> Tuple[str, str]:
        """Extract (class, patientid) from the item."""
        # Support both strings and objects that stringify meaningfully
        s = str(item)
        parts = s.split("-")
        if len(parts) >= 2 and parts[0]:
            cur_class = parts[0]
            cur_patient = parts[1]
        elif len(parts) >= 2:  # class empty like "-PID-..."
            cur_class = "__ALL__"
            cur_patient = parts[1]
        else:
            # No class/patient separator; treat entire token as patient, unknown class
            cur_class = "__ALL__"
            cur_patient = parts[0]
        return cur_class, cur_patient

    # Aggregate items under (class, patient)
    for it in items:
        c, p = parse(it)
        class_to_patient_items[c][p].append(it)

    val_items: List[Any] = []
    train_items: List[Any] = []

    # For each class, pick whole patients into val so item-count ratio ~ val_ratio
    for c, patient_map in class_to_patient_items.items():
        patients = list(patient_map.keys())
        # total items in this class
        total_items_in_class = sum(len(patient_map[p]) for p in patients)
        if total_items_in_class == 0:
            continue

        target_val = int(round(val_ratio * total_items_in_class))

        # Work with (patient, count) pairs
        patient_counts = [(p, len(patient_map[p])) for p in patients]

        # Shuffle to avoid bias on ties, then sort descending by count for a decent greedy
        rng.shuffle(patient_counts)
        patient_counts.sort(key=lambda x: x[1], reverse=True)

        val_patients = set()
        running = 0

        # First pass: add largest patients while not overshooting target
        for p, cnt in patient_counts:
            if running + cnt <= target_val:
                val_patients.add(p)
                running += cnt

        # If we still haven't reached the target, add the patient that minimizes the absolute diff
        if running < target_val:
            candidates = [
                (p, cnt) for p, cnt in patient_counts if p not in val_patients
            ]
            if candidates:
                # Choose the candidate that gets us closest to target
                best_p, best_cnt = min(
                    candidates, key=lambda pc: abs((running + pc[1]) - target_val)
                )
                val_patients.add(best_p)
                running += best_cnt

        # Optional adjustment: if we overshot badly, try removing one patient to get closer
        # (keeps whole-patient integrity)
        if running > target_val and len(val_patients) > 0:
            # Evaluate removing each chosen patient
            chosen_list = [(p, len(patient_map[p])) for p in val_patients]
            best_set = val_patients
            best_diff = abs(running - target_val)
            for p, cnt in chosen_list:
                diff = abs((running - cnt) - target_val)
                if diff < best_diff:
                    # better by removing this patient
                    best_diff = diff
                    best_set = val_patients.copy()
                    best_set.remove(p)
            val_patients = best_set

        # Now collect items
        for p, its in patient_map.items():
            if p in val_patients:
                val_items.extend(its)
            else:
                train_items.extend(its)

    return train_items, val_items


def sample_fraction_per_class(
    items: List[Tuple[str, int, str]], frac: float, rng: random.Random
) -> List[Tuple[str, int, str]]:
    """
    Randomly sample a fraction per class (used for test selection).
    ### NEW/CHANGED ### allows true 0.0 -> 0 samples.
    """
    frac = float(frac)
    by_class: Dict[int, List[Tuple[str, int, str]]] = {}
    for rec in items:
        by_class.setdefault(rec[1], []).append(rec)

    selected = []
    for cls, recs in by_class.items():
        pool = recs[:]
        rng.shuffle(pool)
        if frac <= 0:
            k = 0
        elif frac >= 1:
            k = len(pool)
        else:
            k = int(round(len(pool) * frac))
            if k == 0 and len(pool) > 0:
                k = 1  # at least 1 if fraction > 0
        selected.extend(pool[:k])
    return selected


# ---------------------------
# Dataset
# ---------------------------


class ImageListDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int, str]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, _pid = self.items[idx]
        # Converts grayscale to 3-ch as well
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------
# Models
# ---------------------------


def create_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch.startswith("resnet"):
        fn = getattr(torchvision.models, arch, None)
        if fn is None:
            raise ValueError(f"Unknown torchvision model: {arch}")
        try:
            model = fn(weights="DEFAULT" if pretrained else None)
        except TypeError:
            model = fn(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if arch.startswith("mobilenet_v3"):
        fn = getattr(torchvision.models, arch, None)
        if fn is None:
            raise ValueError(f"Unknown torchvision model: {arch}")
        try:
            model = fn(weights="DEFAULT" if pretrained else None)
        except TypeError:
            model = fn(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # Try timm
    if HAS_TIMM:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
        return model

    raise ValueError(
        f"Unknown model '{arch}'. Install timm for more options (e.g., dinov2_*), "
        f"or use torchvision models like resnet18/resnet50/mobilenet_v3_large."
    )


# ---------------------------
# Training / Evaluation
# ---------------------------


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0) if targets.numel() else 0.0


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels).item()
            total_loss += loss
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


def _interp_from_str(name: str) -> InterpolationMode:
    name = (name or "bicubic").lower()
    return {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
        "lanczos": InterpolationMode.BICUBIC,  # closest available
        "box": InterpolationMode.BILINEAR,
        "hamming": InterpolationMode.BICUBIC,
    }.get(name, InterpolationMode.BICUBIC)


def run_one_training(
    seed: int,
    device: str,
    real_train_items: List[Tuple[str, int, str]],
    real_val_items: List[Tuple[str, int, str]],
    test_items: List[Tuple[str, int, str]],
    syn_train_items: Optional[List[Tuple[str, int, str]]],
    real_mix: float,
    arch: str,
    num_classes: int,
    img_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    pretrained: bool,
    class_weights: List[float],
    log_path: Path,
):
    set_global_seed(seed)

    # Build model first so we can query timm data cfg
    model = create_model(arch, num_classes=num_classes, pretrained=pretrained)
    model.to(device)

    # ### NEW/CHANGED ### Resolve effective input size & normalization
    eff_img_size = img_size
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    interp = InterpolationMode.BICUBIC

    if HAS_TIMM and timm_resolve_data_cfg is not None:
        try:
            cfg = timm_resolve_data_cfg(model)
            inp = cfg.get("input_size", None)  # (3, H, W)
            if isinstance(inp, (list, tuple)) and len(inp) == 3:
                if inp[1] == inp[2]:
                    if inp[1] != img_size:
                        # Override to model-required size (e.g., DINOv2 -> 518)
                        eff_img_size = int(inp[1])
                else:
                    # Non-square: follow height as reference
                    eff_img_size = int(inp[1])
            mean = cfg.get("mean", mean)
            std = cfg.get("std", std)
            interp = _interp_from_str(cfg.get("interpolation", "bicubic"))
        except Exception:
            pass

    # --- OCT-friendly transforms at correct size ---
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=eff_img_size,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                interpolation=interp,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), shear=3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.25, scale=(0.01, 0.05), ratio=(0.3, 3.3), value="random"
            ),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((eff_img_size, eff_img_size), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # --- Build training set by mixing real & synthetic per real_mix ---
    real_mix = max(0.0, min(1.0, float(real_mix)))
    rng = random.Random(seed)

    real_pool = real_train_items[:]
    rng.shuffle(real_pool)
    syn_pool = syn_train_items[:] if syn_train_items is not None else []

    n_real = int(round(len(real_pool) * real_mix))
    n_syn = int(round(len(syn_pool) * (1.0 - real_mix)))
    mixed_train = real_pool[:n_real] + syn_pool[:n_syn]
    rng.shuffle(mixed_train)

    # Datasets + loaders
    train_ds = ImageListDataset(mixed_train, transform=train_tf)
    val_ds = ImageListDataset(
        real_val_items, transform=eval_tf
    )  # validation on REAL holdout
    test_ds = ImageListDataset(test_items, transform=eval_tf)
    # print(len(train_ds), len(val_ds), len(test_ds))  # TODO

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Class-weighted CE Loss
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_acc = 0.0
    best_epoch = -1

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"run_timestamp: {timestamp()}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"arch: {arch}\n")
        f.write(f"effective_img_size: {eff_img_size}\n")  # ### NEW/CHANGED ###
        f.write(f"norm_mean: {mean} | norm_std: {std}\n")  # ### NEW/CHANGED ###
        f.write(
            f"num_train_images (mixed): {len(train_ds)} (real:{n_real} syn:{n_syn})\n"
        )
        f.write(f"num_val_images (real holdout): {len(val_ds)}\n")
        f.write(f"num_test_images: {len(test_ds)}\n")
        f.write(f"epochs: {epochs}, batch_size: {batch_size}\n")
        f.write(f"lr: {lr}, weight_decay: {weight_decay}, pretrained: {pretrained}\n")
        f.write(f"class_weights: {class_weights}\n")
        f.write(f"device: {device}\n")
        f.write("-" * 60 + "\n")
        f.flush()

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for imgs, labels in tqdm(train_loader):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)

            train_loss = train_loss_sum / max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            # Validate
            # print(f"len(val_loader):", len(val_loader))  # TODO
            val_loss, val_acc = evaluate(model, val_loader, device)

            scheduler.step()

            line = (
                f"epoch {epoch:03d} | "
                f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val_loss {val_loss:.4f} acc {val_acc:.4f}"
            )
            print(line)
            f.write(line + "\n")
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

        # Final test evaluation
        test_loss, test_acc = evaluate(model, test_loader, device)

        f.write("-" * 60 + "\n")
        f.write(f"best_val_acc: {best_val_acc:.4f} @ epoch {best_epoch}\n")
        f.write(f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}\n")
        f.flush()


# ---------------------------
# Main / CLI
# ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="OCT classifier with patient split, synthetic mixing, and test eval."
    )

    # Required paths
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="REAL data root with class subfolders 0,1,2,3.",
    )
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Directory to write run .txt logs."
    )

    # Optional paths
    parser.add_argument(
        "--syn-data-dir",
        type=str,
        default=None,
        help="Synthetic data root with class subfolders 0,1,2,3.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Independent test data root with class subfolders 0,1,2,3.",
    )

    # Model / training
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Backbone: resnet18/resnet50/mobilenet_v3_large or timm model (e.g., vit_small_patch14_dinov2.lvd142m).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Base img size; will be overridden to model-required size for some timm models (e.g., DINOv2=518).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for the backbone.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of REAL patients per class used for validation.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test sampling
    parser.add_argument(
        "--test-dir-amount",
        type=float,
        default=1,
        help="Fraction per class from test_dir used for evaluation (default 0.10).",
    )

    # Seeding / repeats
    parser.add_argument(
        "--k-seeds", type=int, default=3, help="How many runs (seeds) to execute."
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="First seed; subsequent seeds are base_seed+i.",
    )

    # Subset options (REAL; mutually exclusive)
    parser.add_argument(
        "--subset-n-per-class",
        type=int,
        default=None,
        help="Use at most N REAL images per class before splitting.",
    )
    parser.add_argument(
        "--subset-percent",
        type=float,
        default=None,
        help="Use P%% of REAL images per class before splitting.",
    )

    # Synthetic subsetting (mutually exclusive)
    parser.add_argument(
        "--syn-subset-n-per-class",
        type=int,
        default=None,
        help="Use at most N SYNTHETIC images per class before mixing.",
    )
    parser.add_argument(
        "--syn-subset-percent",
        type=float,
        default=None,
        help="Use P%% of SYNTHETIC images per class before mixing.",
    )

    # Real/Synthetic mixing
    parser.add_argument(
        "--real-mix",
        type=float,
        default=1.0,
        help="Proportion of REAL samples in training mix (1.0=only real, 0.0=only synthetic).",
    )

    # Class weights
    parser.add_argument(
        "--class-weights",
        type=str,
        default="0.25,0.25,0.25,0.25",
        help="Comma-separated weights for classes 0..3 (e.g. '0.1,0.2,0.3,0.4').",
    )

    args = parser.parse_args()

    classes = ["0", "1", "2", "3"]

    # Prepare run dir
    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    # --- Discover REAL & TEST items ---
    real_root = Path(args.data_dir)
    test_root = Path(args.test_dir)

    all_real_items = discover_images(real_root, classes)
    all_test_items = discover_images(test_root, classes)

    # Apply REAL subsetting before splitting
    rng_fixed_real = random.Random(12345)
    filtered_real = apply_class_subset(
        all_real_items,
        subset_n_per_class=args.subset_n_per_class,
        subset_percent=args.subset_percent,
        rng=rng_fixed_real,
    )

    # Discover + optionally subset SYNTHETIC now (static across seeds), but seed-specific mixing later
    syn_items = None
    if args.syn_data_dir:
        syn_root = Path(args.syn_data_dir)
        all_syn_items = discover_images(syn_root, classes)
        rng_fixed_syn = random.Random(67890)
        syn_items = apply_class_subset(
            all_syn_items,
            subset_n_per_class=args.syn_subset_n_per_class,
            subset_percent=args.syn_subset_percent,
            rng=rng_fixed_syn,
        )

    # Parse class weights
    cw = [float(x.strip()) for x in args.class_weights.split(",")]
    if len(cw) != 4:
        raise ValueError("--class-weights must provide 4 numbers for classes 0..3")

    # Train/eval K seeds
    for i in range(args.k_seeds):
        seed = args.base_seed + i

        split_rng = random.Random(seed)
        real_train_items, real_val_items = stratified_patient_split(
            filtered_real, val_ratio=args.val_ratio, rng=split_rng
        )

        ################################################################################################################
        # Detect patient_id leakage
        def patient_id(item: str) -> str:
            s = str(item)
            parts = s.split("-")
            # assume "<class>-<patient>-..." ; if no '-', treat whole as id
            return parts[1] if len(parts) >= 2 else s

        # Collect patient IDs per split
        train_ids = {patient_id(x) for x in real_train_items}
        val_ids = {patient_id(x) for x in real_val_items}

        # Check for leakage (same patient in both)
        overlap_ids = train_ids & val_ids
        if overlap_ids:
            # Optional: show a few offending examples
            from collections import defaultdict

            examples = defaultdict(lambda: {"train": [], "val": []})
            for it in real_train_items:
                pid = patient_id(it)
                if pid in overlap_ids and len(examples[pid]["train"]) < 3:
                    examples[pid]["train"].append(it)
            for it in real_val_items:
                pid = patient_id(it)
                if pid in overlap_ids and len(examples[pid]["val"]) < 3:
                    examples[pid]["val"].append(it)

            print(
                f"[LEAKAGE] {len(overlap_ids)} patient IDs are in both splits: {sorted(overlap_ids)[:10]}..."
            )
            for pid, ex in list(examples.items())[:3]:
                print(f"  - {pid}: train→{ex['train']}, val→{ex['val']}")
            # raise AssertionError(f"Patient leakage detected: {overlap_ids}")
        else:
            print("✅ No patient overlap between train and val.")

        # Optional: quick sanity stats
        print(
            f"train items: {len(real_train_items)} | val items: {len(real_val_items)}"
        )
        ################################################################################################################

        test_rng = random.Random(seed + 100000)
        test_items = sample_fraction_per_class(
            all_test_items, args.test_dir_amount, test_rng
        )

        unique_name = f"{timestamp()}_{args.arch}_seed{seed}.txt"
        log_path = run_root / unique_name

        run_one_training(
            seed=seed,
            device=args.device,
            real_train_items=real_train_items,
            real_val_items=real_val_items,
            test_items=test_items,
            syn_train_items=syn_items,
            real_mix=args.real_mix,
            arch=args.arch,
            num_classes=len(classes),
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            pretrained=args.pretrained,
            class_weights=cw,
            log_path=log_path,
        )

    print(f"All runs complete. Logs saved under: {run_root.resolve()}")


if __name__ == "__main__":
    main()
