# 1. Train VAE-GAN

This script trains a **VAE-GAN** for OCT-MAISI using an autoencoder (`AutoencoderKlMaisi`).  
It supports mixed precision (AMP), Weights & Biases logging, and saves checkpoints/reconstruction plots.

## Usage

```bash
python scripts/train_vae.py --config ./configs/config_VAE.json
```

# 1.1 Encode Images to Latents

Converts input images to **latent tensors** using a trained `AutoencoderKlMaisi`.  
Reads model hyperparams from `./configs/config_VAE_norm_v1.json`, loads weights, applies transforms, and saves one `.pt` per image.

## Usage

```bash
python scripts/encode_to_latents.py \
  --input_dir /path/to/images \
  --output_dir /path/to/latents \
  --autoencoder_path ./runs/vae_run/model_best.pt
```

# 2. Train Diffusion Model

This script trains the **diffusion model** for OCT-MAISI.  
It handles run directory setup, seed fixing, image discovery/splitting, W&B logging, and kicks off `diff_model_train()`.

## Usage

```bash
python scripts/train_diffusion.py --config ./configs/config_DIFF.json --name DIFFUSION