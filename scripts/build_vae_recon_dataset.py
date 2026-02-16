import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from networks.autoencoderkl_maisi import AutoencoderKlMaisi


def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file()])


def main():
    ap = argparse.ArgumentParser(description="Build VAE reconstruction dataset")
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--deterministic", action="store_true", help="Use z_mu instead of stochastic sample")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--half", action="store_true", help="Run autoencoder in float16")
    args = ap.parse_args()

    real_root = Path(args.real_root)
    out_root = Path(args.out_root)
    vae_ckpt = Path(args.vae_ckpt)

    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(vae_ckpt), map_location="cpu")
    model_cfg = ckpt["config"]["model"]["autoencoder"]
    resize = ckpt["config"]["data"]["val_transform"]["resize"]
    resize_hw = (int(resize[0]), int(resize[1]))

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = AutoencoderKlMaisi(**model_cfg).to(device)
    ae.load_state_dict(ckpt["autoencoder_state_dict"], strict=True)
    ae = ae.half() if args.half else ae.float()
    ae.eval()

    xform = transforms.Compose(
        [
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1),
        ]
    )

    for cls in ["0", "1", "2", "3"]:
        src_dir = real_root / cls
        dst_dir = out_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = list_images(src_dir)
        for i, p in enumerate(files, 1):
            img = Image.open(str(p)).convert("L")
            dtype = torch.float16 if args.half else torch.float32
            x = xform(img).unsqueeze(0).to(device=device, dtype=dtype)
            with torch.no_grad():
                if args.deterministic:
                    z_mu, _ = ae.encode(x)
                    recon = ae.decode(z_mu)
                else:
                    recon, _, _ = ae(x)
            recon = recon.float().clamp(-1.0, 1.0)
            recon_u8 = (
                ((recon + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)[0, 0].cpu().numpy()
            )
            Image.fromarray(recon_u8, mode="L").save(dst_dir / (p.stem + ".png"))
            if i % 200 == 0:
                print(f"class {cls}: {i}/{len(files)}")

    summary = {
        "real_root": str(real_root),
        "out_root": str(out_root),
        "vae_ckpt": str(vae_ckpt),
        "deterministic": bool(args.deterministic),
    }
    (out_root / "meta_recon.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("done", out_root)


if __name__ == "__main__":
    main()
