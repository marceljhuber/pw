import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Ensure repo root is on sys.path when invoked as a script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.utils import define_instance


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def classify_from_filename(path: Path) -> str:
    name = path.name.upper()
    for cls in CLASS_TO_ID:
        if cls in name:
            return cls
    raise ValueError(f"Cannot infer class from filename: {path}")


def count_source_distribution(source_dir: Path) -> dict:
    counts = {c: 0 for c in CLASS_TO_ID}
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        cls = classify_from_filename(p)
        counts[cls] += 1
    return counts


def save_u8_png(img_u8: torch.Tensor, out_path: Path) -> None:
    """Save image tensor [H,W] uint8 as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = img_u8.detach().cpu().numpy()
    Image.fromarray(arr, mode="L").save(out_path)


def load_vae(vae_ckpt_path: Path, device: torch.device) -> AutoencoderKlMaisi:
    ckpt = torch.load(str(vae_ckpt_path), map_location=device)
    model_cfg = ckpt["config"]["model"]["autoencoder"]
    autoencoder = AutoencoderKlMaisi(**model_cfg).to(device)
    autoencoder.load_state_dict(ckpt["autoencoder_state_dict"], strict=True)
    autoencoder.eval()
    return autoencoder


def load_diffusion_and_scheduler(diff_ckpt_path: Path, diff_config_path: Path, device: torch.device):
    with open(diff_config_path, "r") as f:
        cfg = json.load(f)

    merged = {}
    for section in ["main", "model_config", "env_config", "vae_def"]:
        if section in cfg:
            merged.update(cfg[section])
    args = argparse.Namespace(**merged)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    noise_scheduler = define_instance(args, "noise_scheduler")

    ckpt = torch.load(str(diff_ckpt_path), map_location=device)
    diffusion_unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
    diffusion_unet.eval()
    scale_factor = float(ckpt.get("scale_factor", 1.0))
    return diffusion_unet, noise_scheduler, scale_factor


def load_controlnet(controlnet_ckpt_path: Path, controlnet_config_path: Path, device: torch.device):
    with open(controlnet_config_path, "r") as f:
        cfg = json.load(f)

    args = argparse.Namespace(**{})
    for section in ["environment", "model_def", "training"]:
        for k, v in cfg.get(section, {}).items():
            setattr(args, k, v)

    controlnet = define_instance(args, "controlnet_def").to(device)
    ckpt = torch.load(str(controlnet_ckpt_path), map_location=device)
    controlnet.load_state_dict(ckpt["controlnet_state_dict"], strict=True)
    controlnet.eval()
    return controlnet


@torch.inference_mode()
def generate_batch(
    *,
    batch_size: int,
    class_id: int,
    latent_h: int,
    latent_w: int,
    num_classes: int,
    steps: int,
    device: torch.device,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    autoencoder,
    scale_factor: float,
    seed: int,
):
    # Seed per-batch for reproducibility
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    latents = torch.randn(
        (batch_size, 4, latent_h, latent_w),
        generator=g,
        device=device,
        dtype=torch.float16,
    )

    cond_h = latent_h * 4
    cond_w = latent_w * 4
    cond = torch.zeros(
        (batch_size, num_classes, cond_h, cond_w), device=device, dtype=torch.float16
    )
    cond[:, class_id, :, :] = 1.0

    noise_scheduler.set_timesteps(num_inference_steps=int(steps))
    timesteps = noise_scheduler.timesteps

    # Denoise
    with torch.autocast("cuda", enabled=(device.type == "cuda")):
        for t in timesteps:
            t_tensor = torch.full(
                (batch_size,), float(t), device=device, dtype=torch.float32
            )
            down_res, mid_res = controlnet(
                x=latents, timesteps=t_tensor, controlnet_cond=cond
            )
            noise_pred = diffusion_unet(
                x=latents,
                timesteps=t_tensor,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            )
            latents, _ = noise_scheduler.step(noise_pred, t, latents)

        # Decode
        recon = autoencoder.decode_stage_2_outputs(latents / scale_factor)

    # Postprocess to uint8
    # recon is in [-1, 1] (expected), map to [0, 255]
    recon = recon.float().clamp(-1.0, 1.0)
    recon = (recon + 1.0) * 0.5 * 255.0
    recon_u8 = recon.clamp(0, 255).to(torch.uint8)
    # shape [B, 1, H, W]
    return recon_u8


def main():
    ap = argparse.ArgumentParser(description="Generate class-conditional dataset via ControlNet")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--diff_ckpt", required=True)
    ap.add_argument("--diff_config", required=True)
    ap.add_argument("--controlnet_ckpt", required=True)
    ap.add_argument("--controlnet_config", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--latent_h", type=int, default=32)
    ap.add_argument("--latent_w", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--fixed_per_class", type=int, default=None)
    ap.add_argument("--source_dir", type=str, default=None)
    ap.add_argument("--half_dist", action="store_true")
    ap.add_argument("--dataset_name", type=str, default="generated")
    ap.add_argument("--limit_total", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_root / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Targets
    if args.fixed_per_class is None and not args.half_dist:
        raise SystemExit("Provide --fixed_per_class or --half_dist")

    targets = {c: 0 for c in CLASS_NAMES}
    if args.fixed_per_class is not None:
        for c in CLASS_NAMES:
            targets[c] = int(args.fixed_per_class)
    if args.half_dist:
        if not args.source_dir:
            raise SystemExit("--half_dist requires --source_dir")
        dist = count_source_distribution(Path(args.source_dir))
        for c in CLASS_NAMES:
            targets[c] = max(targets[c], dist[c] // 2)

    if args.limit_total is not None:
        # Scale down targets proportionally (rough) if user caps total.
        total = sum(targets.values())
        cap = int(args.limit_total)
        if cap < total and total > 0:
            ratio = cap / float(total)
            for c in CLASS_NAMES:
                targets[c] = max(1, int(targets[c] * ratio))

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": str(out_root),
        "dataset_dir": str(dataset_dir),
        "vae_ckpt": args.vae_ckpt,
        "diff_ckpt": args.diff_ckpt,
        "diff_config": args.diff_config,
        "controlnet_ckpt": args.controlnet_ckpt,
        "controlnet_config": args.controlnet_config,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "latent_h": int(args.latent_h),
        "latent_w": int(args.latent_w),
        "targets": targets,
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_vae(Path(args.vae_ckpt), device)
    diffusion_unet, noise_scheduler, scale_factor = load_diffusion_and_scheduler(
        Path(args.diff_ckpt), Path(args.diff_config), device
    )
    controlnet = load_controlnet(Path(args.controlnet_ckpt), Path(args.controlnet_config), device)

    print("device", device)
    print("scale_factor", scale_factor)
    print("targets", targets)

    global_idx = 0
    for cls_name in CLASS_NAMES:
        class_id = CLASS_TO_ID[cls_name]
        out_dir = dataset_dir / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(out_dir.glob(f"{cls_name}_*.png"))
        start_idx = len(existing)
        target = int(targets[cls_name])
        if start_idx >= target:
            print(f"{cls_name}: already has {start_idx}/{target}, skipping")
            continue

        print(f"{cls_name}: generating {target - start_idx} (have {start_idx})")

        i = start_idx
        while i < target:
            bsz = min(int(args.batch_size), target - i)
            batch_seed = int(args.seed) + global_idx
            recon_u8 = generate_batch(
                batch_size=bsz,
                class_id=class_id,
                latent_h=int(args.latent_h),
                latent_w=int(args.latent_w),
                num_classes=int(args.num_classes),
                steps=int(args.steps),
                device=device,
                diffusion_unet=diffusion_unet,
                controlnet=controlnet,
                noise_scheduler=noise_scheduler,
                autoencoder=autoencoder,
                scale_factor=float(scale_factor),
                seed=batch_seed,
            )

            for j in range(bsz):
                out_path = out_dir / f"{cls_name}_{i + j:06d}.png"
                save_u8_png(recon_u8[j, 0], out_path)

            i += bsz
            global_idx += 1
            if (i - start_idx) % (10 * int(args.batch_size)) == 0:
                print(f"  {cls_name}: {i}/{target}")


if __name__ == "__main__":
    main()
