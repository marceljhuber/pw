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
from networks.conditional_maisi_wrapper import ConditionalMAISIWrapper
from scripts.utils import define_instance


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}


def save_u8_png(img_u8: torch.Tensor, out_path: Path) -> None:
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
    for section in ["main", "conditional_config", "model_config", "env_config", "vae_def"]:
        if section in cfg:
            merged.update(cfg[section])
    args = argparse.Namespace(**merged)

    if bool(getattr(args, "enable_conditional_training", False)):
        diffusion_unet = ConditionalMAISIWrapper(
            config_args=args,
            num_classes=int(getattr(args, "num_classes", 4)),
            class_emb_dim=int(getattr(args, "class_emb_dim", 64)),
            conditioning_method=str(getattr(args, "conditioning_method", "input_concat")),
        ).to(device)
    else:
        diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    noise_scheduler = define_instance(args, "noise_scheduler")

    ckpt = torch.load(str(diff_ckpt_path), map_location=device)
    diffusion_unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
    diffusion_unet.eval()
    scale_factor = float(ckpt.get("scale_factor", 1.0))
    return diffusion_unet, noise_scheduler, scale_factor


def infer_latent_hw_from_config(diff_config_path: Path):
    """Infer latent H/W from pre-encoded latents listed in diffusion config."""
    with open(diff_config_path, "r") as f:
        cfg = json.load(f)

    latents_path = cfg.get("main", {}).get("latents_path")
    if not latents_path:
        return None

    latents_dir = Path(latents_path)
    if not latents_dir.exists() or not latents_dir.is_dir():
        return None

    sample_files = sorted(latents_dir.glob("*_latent.pt"))
    if not sample_files:
        return None

    sample = torch.load(str(sample_files[0]), map_location="cpu")
    if not isinstance(sample, torch.Tensor):
        return None

    # Expected shape is [B, C, H, W] or [C, H, W]
    if sample.ndim == 4:
        return int(sample.shape[-2]), int(sample.shape[-1])
    if sample.ndim == 3:
        return int(sample.shape[-2]), int(sample.shape[-1])
    return None


@torch.inference_mode()
def generate_batch(
    *,
    batch_size: int,
    latent_h: int,
    latent_w: int,
    steps: int,
    device: torch.device,
    diffusion_unet,
    noise_scheduler,
    autoencoder,
    scale_factor: float,
    seed: int,
    class_label=None,
    use_cfg: bool = False,
    guidance_scale: float = 3.0,
    num_classes: int = 4,
):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    latents = torch.randn(
        (batch_size, 4, latent_h, latent_w),
        generator=g,
        device=device,
        dtype=torch.float16,
    )

    noise_scheduler.set_timesteps(num_inference_steps=int(steps))
    timesteps = noise_scheduler.timesteps

    with torch.autocast("cuda", enabled=(device.type == "cuda")):
        for t in timesteps:
            t_tensor = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)
            if use_cfg and (class_label is not None):
                cond_labels = torch.full(
                    (batch_size,), int(class_label), device=device, dtype=torch.long
                )
                uncond_labels = torch.full(
                    (batch_size,), int(num_classes), device=device, dtype=torch.long
                )
                noise_pred_uncond = diffusion_unet(latents, t_tensor, class_labels=uncond_labels)
                noise_pred_cond = diffusion_unet(latents, t_tensor, class_labels=cond_labels)
                noise_pred = noise_pred_uncond + float(guidance_scale) * (
                    noise_pred_cond - noise_pred_uncond
                )
            elif class_label is not None:
                cond_labels = torch.full(
                    (batch_size,), int(class_label), device=device, dtype=torch.long
                )
                noise_pred = diffusion_unet(latents, t_tensor, class_labels=cond_labels)
            else:
                noise_pred = diffusion_unet(latents, t_tensor)
            latents, _ = noise_scheduler.step(noise_pred, t, latents)

        recon = autoencoder.decode_stage_2_outputs(latents / scale_factor)

    recon = recon.float().clamp(-1.0, 1.0)
    recon = (recon + 1.0) * 0.5 * 255.0
    recon_u8 = recon.clamp(0, 255).to(torch.uint8)
    return recon_u8


def main():
    ap = argparse.ArgumentParser(description="Generate dataset via diffusion-only sampling")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dataset_name", default="diffusion_only")
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--diff_ckpt", required=True)
    ap.add_argument("--diff_config", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--latent_h",
        type=int,
        default=None,
        help="Latent height. If omitted, inferred from main.latents_path in --diff_config.",
    )
    ap.add_argument(
        "--latent_w",
        type=int,
        default=None,
        help="Latent width. If omitted, inferred from main.latents_path in --diff_config.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed_per_class", type=int, default=1000)
    ap.add_argument("--class_conditioned", action="store_true")
    ap.add_argument("--use_cfg", action="store_true")
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--num_classes", type=int, default=4)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    dataset_dir = out_root / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_vae(Path(args.vae_ckpt), device)
    diffusion_unet, noise_scheduler, scale_factor = load_diffusion_and_scheduler(
        Path(args.diff_ckpt), Path(args.diff_config), device
    )

    inferred_hw = infer_latent_hw_from_config(Path(args.diff_config))
    if args.latent_h is None or args.latent_w is None:
        if inferred_hw is None:
            raise ValueError(
                "Could not infer latent size from --diff_config main.latents_path. "
                "Please provide --latent_h and --latent_w explicitly."
            )
        latent_h, latent_w = inferred_hw
        print(f"Using inferred latent size from latents_path: {latent_h}x{latent_w}")
    else:
        latent_h, latent_w = int(args.latent_h), int(args.latent_w)
        if inferred_hw is not None and (latent_h, latent_w) != inferred_hw:
            raise ValueError(
                f"Provided latent size ({latent_h}x{latent_w}) does not match "
                f"encoded latents in config ({inferred_hw[0]}x{inferred_hw[1]})."
            )

    note = (
        "Class-conditioned sampling enabled."
        if bool(args.class_conditioned)
        else "This dataset is generated with an unconditional diffusion model. Files are split into class-named folders for convenience only; there is no class conditioning."
    )

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
        "vae_ckpt": args.vae_ckpt,
        "diff_ckpt": args.diff_ckpt,
        "diff_config": args.diff_config,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "latent_h": int(latent_h),
        "latent_w": int(latent_w),
        "fixed_per_class": int(args.fixed_per_class),
        "class_conditioned": bool(args.class_conditioned),
        "use_cfg": bool(args.use_cfg),
        "guidance_scale": float(args.guidance_scale),
        "num_classes": int(args.num_classes),
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    total_per_class = int(args.fixed_per_class)
    global_idx = 0

    for cls in CLASS_NAMES:
        out_dir = dataset_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob(f"{cls}_*.png"))
        start_idx = len(existing)
        if start_idx >= total_per_class:
            print(f"{cls}: already has {start_idx}/{total_per_class}, skipping")
            continue

        print(f"{cls}: generating {total_per_class - start_idx} (have {start_idx})")
        i = start_idx
        while i < total_per_class:
            bsz = min(int(args.batch_size), total_per_class - i)
            batch_seed = int(args.seed) + global_idx
            recon_u8 = generate_batch(
                batch_size=bsz,
                latent_h=latent_h,
                latent_w=latent_w,
                steps=int(args.steps),
                device=device,
                diffusion_unet=diffusion_unet,
                noise_scheduler=noise_scheduler,
                autoencoder=autoencoder,
                scale_factor=float(scale_factor),
                seed=batch_seed,
                class_label=(CLASS_TO_IDX[cls] if bool(args.class_conditioned) else None),
                use_cfg=bool(args.use_cfg),
                guidance_scale=float(args.guidance_scale),
                num_classes=int(args.num_classes),
            )

            for j in range(bsz):
                save_u8_png(recon_u8[j, 0], out_dir / f"{cls}_{i + j:06d}.png")

            i += bsz
            global_idx += 1
            if (i - start_idx) % (10 * int(args.batch_size)) == 0:
                print(f"  {cls}: {i}/{total_per_class}")


if __name__ == "__main__":
    main()
