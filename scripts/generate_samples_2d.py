import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when invoked as a script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from networks.conditional_maisi_wrapper import ConditionalMAISIWrapper
from scripts.sample import (
    ldm_conditional_sample_one_image,
    ldm_conditional_sample_one_image_controlnet,
)
from scripts.utils import define_instance


def _save_tensor_as_png(img_t: torch.Tensor, out_path: Path) -> None:
    """Save a single-channel image tensor as PNG.

    Expects img_t shape [1, H, W] or [H, W] with values in [0, 255].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if img_t.ndim == 3:
        img_t = img_t[0]
    img_u8 = img_t.clamp(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(img_u8, mode="L").save(out_path)


def _load_vae(vae_ckpt_path: Path, device: torch.device) -> AutoencoderKlMaisi:
    ckpt = torch.load(str(vae_ckpt_path), map_location=device)
    model_cfg = ckpt["config"]["model"]["autoencoder"]
    autoencoder = AutoencoderKlMaisi(**model_cfg).to(device)
    autoencoder.load_state_dict(ckpt["autoencoder_state_dict"], strict=True)
    autoencoder.eval()
    return autoencoder


def _load_diffusion(diff_ckpt_path: Path, diff_config_path: Path, device: torch.device):
    with open(diff_config_path, "r") as f:
        cfg = json.load(f)

    # Merge into args like training does.
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


def _load_controlnet(controlnet_ckpt_path: Path, controlnet_config_path: Path, device: torch.device):
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


def main():
    ap = argparse.ArgumentParser(description="Generate 2D samples (diffusion/controlnet)")
    ap.add_argument("--vae_ckpt", type=str, required=True)
    ap.add_argument("--diff_ckpt", type=str, required=True)
    ap.add_argument("--diff_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent_h", type=int, default=32)
    ap.add_argument("--latent_w", type=int, default=32)
    ap.add_argument("--use_controlnet", action="store_true")
    ap.add_argument("--controlnet_ckpt", type=str, default=None)
    ap.add_argument("--controlnet_config", type=str, default=None)
    ap.add_argument("--label", type=int, default=0)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--class_label", type=int, default=None)
    ap.add_argument("--use_cfg", action="store_true")
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_ckpt = Path(args.vae_ckpt)
    diff_ckpt = Path(args.diff_ckpt)
    diff_cfg = Path(args.diff_config)

    autoencoder = _load_vae(vae_ckpt, device)
    diffusion_unet, noise_scheduler, scale_factor = _load_diffusion(
        diff_ckpt, diff_cfg, device
    )

    latent_shape = (4, int(args.latent_h), int(args.latent_w))

    controlnet = None
    cond = None
    if args.use_controlnet:
        if not args.controlnet_ckpt or not args.controlnet_config:
            raise ValueError("--controlnet_ckpt and --controlnet_config are required")
        controlnet = _load_controlnet(
            Path(args.controlnet_ckpt), Path(args.controlnet_config), device
        )

        # Condition map: one-hot class map across the full image plane (H=W=latent*4)
        cond_h = latent_shape[1] * 4
        cond_w = latent_shape[2] * 4
        cond = torch.zeros((1, args.num_classes, cond_h, cond_w), dtype=torch.float16)
        cond[0, int(args.label)] = 1.0

    for i in range(int(args.num)):
        if controlnet is None:
            img = ldm_conditional_sample_one_image(
                autoencoder=autoencoder,
                diffusion_unet=diffusion_unet,
                noise_scheduler=noise_scheduler,
                scale_factor=scale_factor,
                device=device,
                latent_shape=latent_shape,
                noise_factor=1.0,
                num_inference_steps=int(args.steps),
                class_labels=args.class_label,
                use_cfg=bool(args.use_cfg),
                guidance_scale=float(args.guidance_scale),
                num_classes=int(args.num_classes),
            )
            # img is in [-1, 1], convert to [0, 255] for PNG.
            img_u8 = ((img + 1.0) * 0.5 * 255.0).clamp(0, 255)
            _save_tensor_as_png(img_u8[0], out_dir / f"diff_{i:03d}.png")
        else:
            img, _ = ldm_conditional_sample_one_image_controlnet(
                autoencoder=autoencoder,
                diffusion_unet=diffusion_unet,
                controlnet=controlnet,
                noise_scheduler=noise_scheduler,
                scale_factor=scale_factor,
                device=device,
                combine_label_or=cond,
                latent_shape=latent_shape,
                noise_factor=1.0,
                num_inference_steps=int(args.steps),
            )
            _save_tensor_as_png(img[0], out_dir / f"ctrl_label{args.label}_{i:03d}.png")

    print(f"Wrote {args.num} images to {out_dir}")


if __name__ == "__main__":
    main()
