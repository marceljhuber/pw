# MAISI Training Guide

Complete guide for training MAISI models from scratch.

## Prerequisites

### Hardware Requirements

**Minimum**:
- GPU: 16GB VRAM (e.g., Tesla V100, RTX 4090)
- RAM: 32GB system memory
- Storage: 100GB for datasets + outputs

**Recommended**:
- GPU: 32GB VRAM (e.g., V100 32GB, A100 40GB)
- RAM: 64GB+ system memory
- Storage: 500GB SSD for fast I/O

**For Large-Scale Training** (Full 3D volumes):
- Multiple GPUs with 40GB+ VRAM each
- 128GB+ system RAM
- NVMe SSD storage

### Software Requirements

```bash
# Python 3.8+
python --version

# CUDA 11.7+ (check compatibility with your GPU)
nvcc --version

# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda env create -f environment.yml
conda activate maisi
```

### Key Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
monai>=1.3.0
numpy>=1.23.0
pillow>=9.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
wandb>=0.13.0
```

---

## Dataset Preparation

### Directory Structure

Organize your data as follows:

```
data/
├── train/
│   ├── patient001/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   ├── patient002/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Image Format

- **Format**: PNG, JPEG, TIFF, or NIfTI (.nii, .nii.gz)
- **Bit depth**: 8-bit or 16-bit
- **Color**: Grayscale (1 channel)
- **Dimensions**: Consistent within dataset (e.g., 256×256, 512×512)

### Preprocessing

```python
# Example preprocessing script
from PIL import Image
import numpy as np

def preprocess_image(input_path, output_path):
    # Load image
    img = Image.open(input_path).convert('L')  # Grayscale

    # Resize to consistent size
    img = img.resize((256, 256), Image.BILINEAR)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Save
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    img_pil.save(output_path)
```

---

## Stage 1: VAE-GAN Training

### Configuration

Create/edit `configs/config_VAE.json`:

```json
{
  "main": {
    "run_dir": "./runs",
    "jobname": "vae_oct"
  },
  "training": {
    "batch_size": 8,           // Adjust based on GPU memory
    "learning_rate": 1e-4,
    "epochs": 100,
    "num_workers": 8,          // CPU cores for data loading
    "kl_weight": 1e-6,         // KL divergence weight
    "perceptual_weight": 0.1,  // Perceptual loss weight
    "adv_weight": 0.05,        // Adversarial loss weight
    "log_interval": 1,         // Log every N epochs
    "save_interval": 5,        // Save checkpoint every N epochs
    "val_interval": 1,         // Validate every N epochs
    "recon_loss": "l1",        // "l1" or "l2"
    "amp": true,               // Mixed precision training
    "cache": 0.5               // Cache 50% of data in RAM
  },
  "model": {
    "autoencoder": {
      "spatial_dims": 2,
      "in_channels": 1,
      "out_channels": 1,
      "latent_channels": 4,
      "num_channels": [64, 128, 256],
      "num_res_blocks": [2, 2, 2],
      "norm_num_groups": 32,
      "attention_levels": [false, false, false],
      "norm_float16": true,
      "num_splits": 1,          // Increase for larger images
      "dim_split": 1
    },
    "discriminator": {
      "spatial_dims": 2,
      "num_layers_d": 3,
      "channels": 32,
      "in_channels": 1,
      "out_channels": 1,
      "norm": "INSTANCE"
    }
  },
  "data": {
    "image_dir": "/path/to/train/images",
    "train_transform": {
      "resize": [256, 256],
      "random_crop_scale": [0.8, 1.0],
      "random_flip_prob": 0.5,
      "random_rotation_angle": 10,
      "brightness_adjustment": 0.2,
      "contrast_adjustment": 0.2,
      "speckle_noise_std": 0.1
    },
    "val_transform": {
      "resize": [256, 256]
    }
  }
}
```

### Training Command

```bash
python train_vae.py --config ./configs/config_VAE.json
```

### Expected Output

```
Epoch 1/100
Train - Recon Loss: 0.0234, KL Loss: 0.0012, Perceptual: 0.0876, Adv: 0.0234
Val - Recon Loss: 0.0198, KL Loss: 0.0010, Total: 0.0456
Saved checkpoint: runs/vae_oct/model_epoch_1.pt

Epoch 2/100
...

Epoch 50/100
Best model so far! Saved: runs/vae_oct/model_best.pt
```

### Monitoring with W&B

```bash
# View training in browser
wandb login  # First time only
# Training automatically logs to W&B
# View at https://wandb.ai/your-username/your-project
```

### Hyperparameter Tuning

**If reconstruction quality is poor**:
- Increase `perceptual_weight` (0.1 → 0.2)
- Decrease `kl_weight` (1e-6 → 1e-7)
- Train longer (100 → 200 epochs)

**If training is unstable**:
- Decrease `learning_rate` (1e-4 → 5e-5)
- Decrease `adv_weight` (0.05 → 0.01)
- Increase `batch_size` (8 → 16)

**If memory issues**:
- Decrease `batch_size` (8 → 4)
- Increase `num_splits` (1 → 2 or 4)
- Set `cache=0`

---

## Stage 1.5: Encode Images to Latents (Optional)

### Why Encode?

Pre-encoding images to latent space:
- **Speeds up** diffusion training (no repeated VAE forward passes)
- **Reduces memory** during diffusion training
- **Enables** training on larger datasets

### Command

```bash
python scripts/encode_to_latents.py \
  --input_dir /path/to/train/images \
  --output_dir /path/to/latents/train \
  --autoencoder_path ./runs/vae_oct/model_best.pt \
  --batch_size 16 \
  --num_workers 8
```

### Expected Output

```
Processing: 100%|████████| 10000/10000 [12:34<00:00, 13.24it/s]
Encoded 10000 images to latents
Output directory: /path/to/latents/train
Average latent shape: (4, 32, 32)
```

---

## Stage 2: Diffusion Model Training

### Configuration

Create/edit `configs/config_DIFF.json`:

```json
{
  "main": {
    "enable_conditional_training": true,
    "use_cfg": true,                        // Classifier-free guidance
    "image_dir": "/path/to/train/images",   // If using images directly
    "latents_path": "/path/to/latents/train",  // If using pre-encoded latents
    "trained_autoencoder_path": "./runs/vae_oct/model_best.pt",
    "trained_unet_path": null               // For resuming training
  },
  "conditional_config": {
    "num_classes": 4,                       // Number of condition classes
    "class_emb_dim": 64,
    "conditioning_method": "input_concat"
  },
  "model_config": {
    "diffusion_unet_train": {
      "batch_size": 8,
      "cache_rate": 0.5,                    // Cache data in RAM
      "lr": 0.0001,
      "n_epochs": 1000
    },
    "diffusion_unet_inference": {
      "dim": [256, 256, 128],               // Output dimensions
      "spacing": [1.0, 1.0, 1.0],          // Voxel spacing
      "num_inference_steps": 1000           // DDPM steps (or 50 for DDIM)
    }
  },
  "vae_def": {
    "diffusion_unet_def": {
      "spatial_dims": 2,
      "in_channels": 4,                     // Latent channels from VAE
      "out_channels": 4,
      "num_channels": [64, 128, 256, 512],
      "attention_levels": [false, false, true, true],
      "num_head_channels": [0, 0, 32, 32],
      "num_res_blocks": 2,
      "use_flash_attention": true
    },
    "noise_scheduler": {
      "_target_": "monai.networks.schedulers.ddpm.DDPMScheduler",
      "num_train_timesteps": 1000,
      "beta_start": 0.0015,
      "beta_end": 0.0195,
      "schedule": "scaled_linear_beta",
      "clip_sample": false
    }
  }
}
```

### Training Command

```bash
python train_diffusion.py
```

### Expected Output

```
Epoch 1/1000
Step 100/1250 - Loss: 0.1234
Step 200/1250 - Loss: 0.0987
...
Validation - Generating samples...
Saved samples: runs/diffusion/samples_epoch_1/
Saved checkpoint: runs/diffusion/model_epoch_1.pt

Epoch 50/1000
Loss has improved! Saved: runs/diffusion/model_best.pt
```

### Conditioning Setup

For conditional training, ensure your data includes labels:

```python
# Example: Modify data loading to include conditions
class ConditionalDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels  # e.g., [0, 1, 2, 3] for 4 classes

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        label = self.labels[idx]
        return {
            'image': image,
            'label': label
        }
```

### Monitoring Progress

**Check sample quality**:
```bash
# View generated samples during training
ls runs/diffusion/samples_epoch_*/
```

**Monitor loss curves**:
- W&B dashboard shows real-time training loss
- Look for steady decrease in L1 loss
- Validation samples should improve in quality over time

### Training Tips

**For better sample quality**:
- Train longer (1000+ epochs)
- Use classifier-free guidance (`use_cfg=true`)
- Increase U-Net capacity (more channels/blocks)

**For faster training**:
- Use pre-encoded latents
- Enable mixed precision (`amp=true`)
- Use DDIM scheduler (fewer inference steps)

**If training is slow**:
- Increase `cache_rate` (0.5 → 1.0) if RAM allows
- Use multiple GPUs (add DDP support)
- Reduce `num_inference_steps` during validation

---

## Stage 3: ControlNet Training

### Configuration

Create/edit `configs/config_CONTROLNET_modality.json`:

```json
{
  "main": {
    "trained_autoencoder_path": "./runs/vae_oct/model_best.pt",
    "trained_unet_path": "./runs/diffusion/model_best.pt",
    "controlnet_checkpoint": null,          // For resuming
    "conditioning_type": "modality"         // or "segmentation", "tumor", etc.
  },
  "training": {
    "batch_size": 4,                        // Smaller than diffusion (ControlNet + U-Net)
    "learning_rate": 1e-5,                  // Lower than diffusion
    "epochs": 500,
    "warmup_epochs": 10
  },
  "data": {
    "image_dir": "/path/to/train/images",
    "condition_dir": "/path/to/conditions", // Masks, modality labels, etc.
    "conditioning_format": "image"          // or "mask", "label"
  },
  "vae_def": {
    "controlnet_def": {
      "spatial_dims": 2,
      "in_channels": 4,
      "num_channels": [64, 128, 256, 512],
      "attention_levels": [false, false, true, true],
      "num_head_channels": [0, 0, 32, 32],
      "num_res_blocks": 2,
      "use_flash_attention": true,
      "conditioning_embedding_in_channels": 8,
      "conditioning_embedding_num_channels": [8, 32, 64]
    }
  }
}
```

### Training Command

```bash
# For modality conditioning
python train_controlnet_modality.py

# For RETOUCH pathology conditioning
python train_controlnet_retouch.py
```

### Conditioning Data Preparation

**Modality Conditioning**:
```python
# Create modality labels file
# Format: image_path,modality_id
data/train/patient001/image_001.png,0
data/train/patient001/image_002.png,1
...
```

**Segmentation Mask Conditioning**:
```python
# Prepare segmentation masks
# Match each image with corresponding mask
data/
├── images/
│   └── image_001.png
└── masks/
    └── image_001_mask.png
```

### Expected Output

```
ControlNet Training
Epoch 1/500
Diffusion model: FROZEN ✓
ControlNet: TRAINABLE ✓
Step 100/800 - Control Loss: 0.0876
...
Validation - Generating controlled samples...
Saved: runs/controlnet_modality/samples_epoch_1/
```

### Validation

ControlNet validation should show:
1. **Unconditional samples**: Similar to base diffusion model
2. **Conditional samples**: Follow the conditioning signal
3. **Interpolation**: Gradual transition between conditions

---

## Training Best Practices

### Learning Rate Scheduling

```python
# Warmup + Cosine Decay (recommended)
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def warmup_lambda(epoch, warmup_epochs=10):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: warmup_lambda(e))
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
```

### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Checkpointing Strategy

```python
# Save multiple types of checkpoints
# 1. Best model (lowest validation loss)
if val_loss < best_val_loss:
    save_checkpoint('model_best.pt')

# 2. Periodic checkpoints (every N epochs)
if epoch % save_interval == 0:
    save_checkpoint(f'model_epoch_{epoch}.pt')

# 3. Latest checkpoint (for resuming)
save_checkpoint('model_latest.pt')
```

### Data Augmentation Guidelines

**VAE Training**:
- Heavy augmentation (flip, rotation, crop, intensity)
- Helps VAE generalize to variations

**Diffusion Training**:
- Light augmentation (flip, small crops)
- Preserve data distribution for accurate modeling

**ControlNet Training**:
- Minimal augmentation
- Must preserve conditioning alignment

---

## Troubleshooting

### VAE Issues

**Problem**: Blurry reconstructions
```
Solution:
- Increase perceptual_weight (0.1 → 0.3)
- Use L1 loss instead of L2
- Train longer
```

**Problem**: Mode collapse (all outputs similar)
```
Solution:
- Decrease adv_weight (0.05 → 0.01)
- Increase KL_weight (1e-6 → 1e-5)
- Check discriminator isn't too strong
```

**Problem**: Training unstable (loss oscillates)
```
Solution:
- Decrease learning rate
- Add gradient clipping
- Use smaller batch size
```

### Diffusion Issues

**Problem**: Poor sample quality
```
Solution:
- Train longer (>500 epochs)
- Increase model capacity
- Check VAE quality first
- Use classifier-free guidance
```

**Problem**: Samples don't match conditions
```
Solution:
- Increase condition embedding dimension
- Check condition preprocessing
- Verify labels are correct
- Use higher CFG scale during inference
```

**Problem**: Out of memory
```
Solution:
- Use pre-encoded latents
- Decrease batch_size
- Enable gradient checkpointing
- Reduce model size
```

### ControlNet Issues

**Problem**: ControlNet has no effect
```
Solution:
- Check zero convolutions initialized correctly
- Increase training epochs
- Verify conditioning input is preprocessed correctly
- Use higher control scale during inference
```

**Problem**: ControlNet destroys image quality
```
Solution:
- Decrease control scale
- Train longer with frozen diffusion model
- Check conditioning encoder architecture
```

---

## Hardware-Specific Optimizations

### Single GPU (16GB)

```python
# config adjustments
{
  "batch_size": 4,
  "num_splits": 2,  # TSP for VAE
  "cache_rate": 0,  # No caching
  "amp": true,      # Mixed precision
  "gradient_checkpointing": true
}
```

### Multi-GPU (DDP)

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train_diffusion.py
```

```python
# In code
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

### CPU Training (Not Recommended)

```python
# Extremely slow, only for testing
device = torch.device("cpu")
# Disable AMP, reduce batch size to 1
```

---

## Experiment Tracking

### W&B Setup

```python
import wandb

wandb.init(
    project="maisi-oct",
    name="vae-baseline",
    config={
        "learning_rate": 1e-4,
        "batch_size": 8,
        "epochs": 100
    }
)

# Log during training
wandb.log({
    "train_loss": loss.item(),
    "val_loss": val_loss,
    "epoch": epoch
})

# Log images
wandb.log({"reconstructions": wandb.Image(img)})
```

### TensorBoard (Alternative)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_images('Images/reconstructions', imgs, epoch)
writer.close()
```

---

## Next Steps

After training:
- See [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md) for generation
- See [01_ARCHITECTURE.md](01_ARCHITECTURE.md) for model details
- Experiment with different conditioning strategies
- Fine-tune on domain-specific data
