# Quick Start - Fast Training (64x64)

This guide will help you quickly test the full MAISI pipeline with small 64x64 images for rapid prototyping and validation.

## Setup Environment

```bash
# Option 1: Run setup script (recommended)
chmod +x setup_fast_env.sh
./setup_fast_env.sh

# Option 2: Manual setup
conda activate maisi
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install monai==1.3.0 diffusers==0.25.0 albumentations wandb tensorboard lpips einops
```

## Fast Training Pipeline

### Step 1: Train VAE (5 epochs, ~10-15 min)

```bash
conda activate maisi
python train_vae.py --config ./configs/config_VAE_fast.json
```

**What this does:**
- Trains VAE on 64x64 images
- 5 epochs with batch size 16
- Creates latent compression model
- Saves to: `./runs/VAE/vae_fast64_best.pt`

### Step 2: Encode Images to Latents (~5 min)

```bash
python scripts/encode_to_latents.py \
    --autoencoder_path ./runs/VAE/vae_fast64_best.pt \
    --image_dir /home/user/Thesis/data/KermanyV3_resized/train \
    --output_dir ./latents/KermanyV3_resized_fast/train \
    --image_size 64 \
    --batch_size 64
```

**What this does:**
- Encodes all training images to latent space
- Creates 16x16x4 latent representations from 64x64 images
- Saves latents for diffusion training

### Step 3: Train Diffusion Model (10 epochs, ~20-30 min)

```bash
python train_diffusion.py \
    --config ./configs/config_DIFF_fast.json \
    --name diffusion_fast
```

**What this does:**
- Trains diffusion model on latent space
- 10 epochs with batch size 32
- Learns to generate latent representations
- Saves to: `./runs/DIFFUSION/diffusion_fast_best.pt`

### Step 4: Train ControlNet (10 epochs, ~20-30 min)

```bash
python train_controlnet_modality.py \
    --config ./configs/config_CONTROLNET_fast.json
```

**What this does:**
- Trains ControlNet for class-conditional generation
- Conditions on 4 retinal disease classes
- 10 epochs with batch size 16
- Saves to: `./runs/CONTROLNET/controlnet_fast_best.pt`

### Step 5: Generate Images

```bash
python inference_controlnet.py \
    --config ./configs/config_CONTROLNET_fast.json \
    --num_samples 10 \
    --guidance_scale 3.0 \
    --num_inference_steps 50
```

**What this does:**
- Generates 10 synthetic 64x64 OCT images
- Uses class-conditional ControlNet
- 50 diffusion steps for fast generation
- Saves to: `./outputs/controlnet_fast/`

## Expected Timeline

| Stage | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| VAE Training | ~10-15 min | ~1-2 hours |
| Latent Encoding | ~5 min | ~15-20 min |
| Diffusion Training | ~20-30 min | ~2-3 hours |
| ControlNet Training | ~20-30 min | ~2-3 hours |
| Inference (10 images) | ~30 sec | ~2-3 min |
| **Total** | **~1-1.5 hours** | **~6-8 hours** |

## Configuration Details

### VAE (config_VAE_fast.json)
- Image size: 64x64
- Latent channels: 4
- Model channels: [32, 64, 128]
- Batch size: 16
- Epochs: 5

### Diffusion (config_DIFF_fast.json)
- Latent size: 16x16x4
- Model channels: [32, 64, 128, 256]
- Batch size: 32
- Epochs: 10
- Inference steps: 50 (fast DDIM)

### ControlNet (config_CONTROLNET_fast.json)
- Conditioning: 4 disease classes
- Model channels: [32, 64, 128, 256]
- Batch size: 16
- Epochs: 10

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch sizes in configs
# VAE: batch_size: 8
# Diffusion: batch_size: 16
# ControlNet: batch_size: 8
```

### CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, check drivers
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training
```bash
# Use smaller models - edit configs:
# - num_channels: [16, 32, 64]  # Even smaller
# - num_res_blocks: [1, 1, 1]   # Fewer blocks
# - batch_size: 32               # Larger batches
```

## Next Steps

After validating the pipeline with 64x64 images, scale up to production:

1. **128x128**: Double image size, adjust configs
2. **256x256**: Production quality, longer training
3. **512x512**: High resolution (original paper size)

See [04_TRAINING_GUIDE.md](ai-doc/04_TRAINING_GUIDE.md) for full-scale training.
