import json
import os
import sys

# Ensure repo root is on sys.path so `scripts.*` imports work when invoked directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.utils_data import list_image_files, set_random_seeds, setup_transforms


def process_images(
    input_dir,
    output_dir,
    autoencoder_path,
    vae_config_path,
    skip_existing=True,
    seed=42,
    image_size=None,
    max_images=None,
    subset_patient_fraction=None,
    patient_list_path=None,
):
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE config (ensures model + resize match the checkpoint)
    with open(vae_config_path, "r") as f:
        config = json.load(f)

    if image_size is not None:
        # Override resize in transforms to match requested size.
        config.setdefault("data", {})
        config["data"].setdefault("train_transform", {})
        config["data"].setdefault("val_transform", {})
        config["data"]["train_transform"]["resize"] = [int(image_size), int(image_size)]
        config["data"]["val_transform"]["resize"] = [int(image_size), int(image_size)]

    _, transforms = setup_transforms(config)

    model_config = config["model"]["autoencoder"]

    # Load model
    autoencoder = AutoencoderKlMaisi(**model_config).to(device)
    checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    autoencoder.eval()

    # Create output dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    files = list_image_files(input_dir)

    if patient_list_path is not None:
        p = Path(patient_list_path)
        if not p.exists():
            raise FileNotFoundError(f"patient_list_path not found: {p}")
        keep_pids = {
            line.strip()
            for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if not keep_pids:
            raise ValueError(f"patient_list_path is empty: {p}")

        filtered = []
        for fp in files:
            name = Path(fp).name
            parts = name.split("-")
            pid = parts[1] if len(parts) >= 3 else Path(fp).stem
            if pid in keep_pids:
                filtered.append(fp)
        files = filtered

    if subset_patient_fraction is not None:
        frac = float(subset_patient_fraction)
        if 0 < frac < 1.0:
            patient_to_files = {}
            for p in files:
                name = Path(p).name
                parts = name.split("-")
                pid = parts[1] if len(parts) >= 3 else Path(p).stem
                patient_to_files.setdefault(pid, []).append(p)
            all_pids = sorted(patient_to_files.keys())
            rng = torch.Generator().manual_seed(int(seed))
            # Deterministic shuffle using torch to avoid platform RNG differences
            perm = torch.randperm(len(all_pids), generator=rng).tolist()
            all_pids = [all_pids[i] for i in perm]
            k = max(1, int(round(len(all_pids) * frac)))
            keep_pids = all_pids[:k]
            files = []
            for pid in keep_pids:
                files.extend(patient_to_files[pid])

    if max_images is not None:
        files = files[: int(max_images)]
    with tqdm(files, desc="Converting images to latents") as pbar:
        for filepath in pbar:
            # Convert string path to Path object
            filepath = Path(filepath)
            out_filename = out_dir / f"{filepath.stem}_latent.pt"

            if skip_existing and out_filename.exists():
                continue

            pbar.set_description(f"Processing {filepath.name}")

            image = Image.open(str(filepath)).convert("L")
            image = transforms(image)

            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        latent, _ = autoencoder.encode(image.unsqueeze(0).to(device))
                else:
                    latent, _ = autoencoder.encode(image.unsqueeze(0).to(device))
                torch.save(latent.cpu(), out_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument(
        "--vae_config",
        type=str,
        default="./configs/config_VAE_fast.json",
        help="Path to the VAE config used to define the autoencoder and transforms.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Override resize in transforms (e.g., 64).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap for quick runs.",
    )
    parser.add_argument(
        "--subset_patient_fraction",
        type=float,
        default=None,
        help="Optional fraction of patients to encode (e.g., 0.1).",
    )
    parser.add_argument(
        "--patient_list_path",
        type=str,
        default=None,
        help="Optional text file with one patient id per line to encode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_skip", action="store_false", dest="skip_existing")
    args = parser.parse_args()

    process_images(
        args.input_dir,
        args.output_dir,
        args.autoencoder_path,
        args.vae_config,
        args.skip_existing,
        args.seed,
        args.image_size,
        args.max_images,
        args.subset_patient_fraction,
        args.patient_list_path,
    )
