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
python train_diffusion.py --conf
```

# 3. Train ControlNet Model (Modality)

This script trains the **ControlNet model** (modality) for OCT-MAISI.  
It handles run directory setup, seed fixing, image discovery/splitting, W&B logging, and kicks off `diff_model_train()`.

## Usage

```bash
python train_controlnet_modality.py
```

# 4. Train ControlNet Model (RETOUCH)

This script trains the **ControlNet model** (RETOUCH) for OCT-MAISI.  
It handles run directory setup, seed fixing, image discovery/splitting, W&B logging, and kicks off `diff_model_train()`.

## Usage

```bash
python train_controlnet_retouch.py
```