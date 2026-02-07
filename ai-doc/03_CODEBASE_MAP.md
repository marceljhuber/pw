# MAISI Codebase Map

This document provides a comprehensive guide to the repository structure and key files.

## Repository Structure

```
pw/
├── ai-doc/                      # AI agent documentation (this directory)
├── configs/                     # Configuration files for training/inference
├── networks/                    # Model architectures
├── scripts/                     # Utility functions and training logic
├── runs/                        # Training outputs, checkpoints, logs
├── wandb/                       # Weights & Biases experiment tracking
├── train_*.py                   # Training entry points
├── inference*.py                # Inference entry points
├── *.ipynb                      # Jupyter notebooks for exploration
└── requirements.txt             # Python dependencies
```

## Core Training Scripts

### train_vae.py
**Purpose**: Train the VAE-GAN volume compression network (Stage 1)

**Key Functions**:
- `setup_models()`: Initialize AutoencoderKlMaisi and PatchDiscriminator
- `setup_optimizers()`: Configure Adam optimizers with learning rate scheduling
- `setup_losses()`: Initialize reconstruction, perceptual, and adversarial losses
- `train_step()`: Generator and discriminator training step
- `validate()`: Validation loop with visualization

**Configuration**: `configs/config_VAE.json`

**Usage**:
```bash
python train_vae.py --config ./configs/config_VAE.json
```

**Outputs**:
- `runs/{jobname}/`: Model checkpoints, logs, visualizations
- `runs/{jobname}/model_best.pt`: Best model based on validation loss
- `runs/{jobname}/model_epoch_{n}.pt`: Periodic checkpoints

**Key Features**:
- Mixed precision training (AMP) for efficiency
- KL divergence regularization with clamped std (0.9-1.1)
- Perceptual loss using SqueezeNet
- PatchGAN discriminator (3 layers)
- Data augmentation: random crop, flip, rotation, intensity scaling
- Weights & Biases logging

---

### train_diffusion.py
**Purpose**: Train the latent diffusion model (Stage 2)

**Key Functions**:
- Calls `scripts/diff_model_train.py` for actual training
- Sets up run directory and configuration
- Handles seed setting for reproducibility
- Discovers and splits image files into train/val

**Configuration**: `configs/config_DIFF.json`

**Usage**:
```bash
python train_diffusion.py
```

**Workflow**:
1. Reads configuration from config file
2. Sets up output directory (`runs/diffusion/`)
3. Discovers image files (or latent files if pre-encoded)
4. Splits into train/validation sets
5. Calls `diff_model_train()` from scripts

**Dependencies**:
- Trained VAE from Stage 1 (`trained_autoencoder_path`)
- Optionally: Pre-encoded latents (`latents_path`)

---

### train_controlnet_modality.py
**Purpose**: Train ControlNet for modality conditioning (Stage 3)

**Target**: OCT imaging modality control

**Configuration**: `configs/config_CONTROLNET_modality.json`

**Usage**:
```bash
python train_controlnet_modality.py
```

**Key Aspects**:
- Freezes diffusion model weights
- Trains only ControlNet parameters
- Modality-specific conditioning encoder
- Zero convolution initialization

---

### train_controlnet_retouch.py
**Purpose**: Train ControlNet for RETOUCH retinal pathology

**Target**: Retinal fluid segmentation conditioning

**Configuration**: `configs/config_CONTROLNET_retouch.json`

**Usage**:
```bash
python train_controlnet_retouch.py
```

**Differences from modality**:
- Different conditioning input (segmentation masks)
- RETOUCH dataset-specific preprocessing

---

### train_classifier.py
**Purpose**: Train a classifier for conditional generation

**Note**: Auxiliary script for classification-based conditioning

**Configuration**: Custom (see file header)

---

## Core Inference Scripts

### inference.py
**Purpose**: Basic diffusion model inference

**Features**:
- Load trained VAE and diffusion model
- Generate samples with body region and voxel spacing conditioning
- Save generated images

**Configuration**: `configs/config_INFERENCE_*.json`

**Usage**:
```bash
python inference.py --config ./configs/config_INFERENCE_v1.json
```

---

### inference_controlnet.py
**Purpose**: Generate images using ControlNet

**Features**:
- Load trained VAE, diffusion model, and ControlNet
- Conditional generation with task-specific inputs
- Supports multiple conditioning types

**Configuration**: `configs/config_CONTROLNET_*.json`

**Usage**:
```bash
python inference_controlnet.py --config ./configs/config_CONTROLNET_germany.json
```

---

### inference_optimized.py
**Purpose**: Optimized inference with DDIM sampling

**Features**:
- Faster sampling with fewer timesteps
- Memory-optimized for large volumes
- Optional classifier-free guidance

**Usage**:
```bash
python inference_optimized.py
```

---

## Network Architectures

### networks/autoencoderkl_maisi.py

**Class**: `AutoencoderKlMaisi`

**Inheritance**: Extends `monai.networks.nets.AutoencoderKL`

**Key Components**:

#### MaisiGroupNorm3D
Custom group normalization with:
- Optional FP16 conversion
- Memory optimization (CUDA cache clearing)
- Configurable print_info for debugging

#### AutoencoderKlMaisi
Enhanced VAE with:
- **Tensor Splitting Parallelism**: `num_splits`, `dim_split`
- **Memory optimization**: `save_mem` flag
- **Mixed precision**: `norm_float16`
- **Encoder/Decoder**: Symmetric architecture with residual blocks

**Key Methods**:
- `encode()`: Image → latent features
- `decode()`: Latent features → reconstructed image
- `forward()`: Full encode-decode with KL loss
- `encode_stage_2_inputs()`: Prepare latents for diffusion model

**Configuration Example**:
```python
autoencoder = AutoencoderKlMaisi(
    spatial_dims=2,           # 2D or 3D
    in_channels=1,            # Grayscale
    out_channels=1,
    latent_channels=4,        # Compression factor
    num_channels=[64,128,256],# Encoder/decoder channels
    num_res_blocks=[2,2,2],   # Residual blocks per stage
    norm_num_groups=32,       # GroupNorm groups
    attention_levels=[False,False,False],
    norm_float16=True,        # FP16 optimization
    num_splits=8,             # TSP splits
    dim_split=1               # Split dimension
)
```

---

### networks/controlnet_maisi.py

**Class**: `ControlNetMaisi`

**Purpose**: Provide conditional control for diffusion generation

**Architecture**:
- Mirrors diffusion U-Net structure
- Additional conditioning encoder
- Zero convolution layers for gradual integration

**Key Components**:

#### Conditioning Encoder
```python
conditioning_embedding_in_channels=8,     # Input channels (e.g., mask channels)
conditioning_embedding_num_channels=[8,32,64]  # Encoder progression
```

#### ControlNet Blocks
- Identical to U-Net structure
- Initialized from pre-trained U-Net weights
- Connected via zero convolutions

**Key Methods**:
- `forward()`: Process conditioning and return control signals
- `zero_convs`: List of zero-initialized convolutions

**Usage**:
```python
controlnet = ControlNetMaisi(
    spatial_dims=2,
    in_channels=4,            # Latent channels
    num_channels=[64,128,256,512],
    attention_levels=[False,False,True,True],
    num_head_channels=[0,0,32,32],
    num_res_blocks=2,
    conditioning_embedding_in_channels=8,
    conditioning_embedding_num_channels=[8,32,64]
)
```

---

### networks/conditional_maisi_wrapper.py

**Class**: `ConditionalMaisiWrapper`

**Purpose**: Wrapper for conditional diffusion model training

**Features**:
- Integrates U-Net with conditioning mechanisms
- Handles classifier-free guidance
- Manages conditional/unconditional forward passes

---

### networks/schedulers/

Noise scheduling algorithms for diffusion:

#### ddim.py
**DDIM Scheduler**: Deterministic sampling for faster inference

#### ddim_hacked.py
**Modified DDIM**: Custom variations for experimentation

#### scheduler.py
**Base Scheduler**: Abstract scheduler class

#### utils.py / util.py
Utility functions for noise scheduling

---

## Scripts Directory

### scripts/diff_model_train.py

**Function**: `diff_model_train()`

**Purpose**: Core training loop for diffusion model

**Key Steps**:
1. Load pre-trained VAE (frozen)
2. Initialize diffusion U-Net
3. Set up noise scheduler (DDPM/DDIM)
4. Training loop:
   - Encode images to latents (if needed)
   - Add noise to latents
   - Predict noise with U-Net
   - Compute L1 loss
   - Backpropagate and update

**Features**:
- Conditional training (body region, voxel spacing)
- Classifier-free guidance training
- Mixed precision (AMP)
- Gradient checkpointing
- Validation with image generation

---

### scripts/diff_model_infer.py

**Function**: `diff_model_infer()`

**Purpose**: Inference logic for diffusion model

**Process**:
1. Load trained VAE and U-Net
2. Sample random noise
3. Iterative denoising (T steps)
4. Decode latents to images
5. Save generated samples

**Options**:
- DDPM (1000 steps) vs DDIM (50-100 steps)
- Classifier-free guidance scale
- Conditional generation

---

### scripts/diff_model_setting.py

**Functions**: Configuration and setup utilities

---

### scripts/train_controlnet.py

**Function**: `train_controlnet()`

**Purpose**: Training loop for ControlNet

**Key Differences from Diffusion Training**:
- Loads pre-trained diffusion model (frozen)
- Initializes ControlNet from diffusion weights
- Adds conditioning encoder
- Only updates ControlNet parameters

---

### scripts/infer_controlnet.py

**Function**: `infer_controlnet()`

**Purpose**: Inference with ControlNet conditioning

**Process**:
1. Load VAE, diffusion model, ControlNet
2. Prepare conditioning input (mask, image, etc.)
3. Encode conditioning
4. Diffusion sampling with control signals
5. Decode and save

---

### scripts/encode_to_latents.py

**Purpose**: Pre-encode images to latent space

**Usage**:
```bash
python scripts/encode_to_latents.py \
  --input_dir /path/to/images \
  --output_dir /path/to/latents \
  --autoencoder_path ./runs/vae/model_best.pt
```

**Benefits**:
- Faster diffusion training (no repeated encoding)
- Reduced memory usage
- Saves latents as `.pt` files

---

### scripts/utils.py

**Key Functions**:

#### `KL_loss(mu, logvar)`
KL divergence for VAE regularization with std clamping

#### `setup_device()`
Configure GPU/CPU device

#### `save_checkpoint()`
Save model state with metadata

#### `load_checkpoint()`
Load model state from checkpoint

---

### scripts/utils_data.py

**Key Functions**:

#### `set_random_seeds(seed)`
Reproducibility: sets Python, NumPy, PyTorch seeds

#### `list_image_files(directory, extensions)`
Recursively find images

#### `split_train_val_by_patient(files, val_ratio)`
Patient-aware train/validation split

#### `setup_transforms(config)`
Data augmentation pipeline:
- Random crop with scale
- Random flip
- Random rotation
- Brightness/contrast adjustment
- Speckle noise (OCT-specific)

#### `setup_dataloaders(train_files, val_files, config)`
Create PyTorch DataLoaders with caching

---

### scripts/utils_plot.py

**Key Functions**:

#### `visualize_2d(images, titles, save_path)`
Plot multiple 2D images in grid

#### `plot_training_curves(metrics, save_path)`
Visualize loss curves

---

### scripts/controlnet_utils.py

**Functions**: Utilities specific to ControlNet training/inference

---

### scripts/sample.py

**Functions**: Sampling utilities for diffusion models

---

## Configuration Files

### configs/config_VAE.json

VAE-GAN training configuration

**Key Sections**:
```json
{
  "main": {
    "run_dir": "./runs",
    "jobname": "vae"
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs": 100,
    "kl_weight": 1e-6,
    "perceptual_weight": 0.1,
    "adv_weight": 0.05,
    "amp": true
  },
  "model": {
    "autoencoder": { ... },
    "discriminator": { ... }
  },
  "data": {
    "image_dir": "/path/to/images",
    "train_transform": { ... },
    "val_transform": { ... }
  }
}
```

---

### configs/config_DIFF.json

Diffusion model training configuration

**Key Sections**:
```json
{
  "main": {
    "enable_conditional_training": true,
    "use_cfg": true,
    "latents_path": "/path/to/latents",
    "trained_autoencoder_path": "./runs/vae/model_best.pt"
  },
  "conditional_config": {
    "num_classes": 4,
    "class_emb_dim": 64,
    "conditioning_method": "input_concat"
  },
  "model_config": {
    "diffusion_unet_train": {
      "batch_size": 8,
      "lr": 0.0001,
      "n_epochs": 1000
    }
  },
  "vae_def": {
    "diffusion_unet_def": { ... },
    "noise_scheduler": { ... }
  }
}
```

---

### configs/config_CONTROLNET_*.json

ControlNet configurations for different tasks

**Variants**:
- `config_CONTROLNET_v1.json`: Basic ControlNet
- `config_CONTROLNET_v2.json`: Improved version
- `config_CONTROLNET_modality.json`: Modality conditioning
- `config_CONTROLNET_canada.json`: Canada dataset
- `config_CONTROLNET_denmark.json`: Denmark dataset
- `config_CONTROLNET_england.json`: England dataset
- `config_CONTROLNET_france.json`: France dataset
- `config_CONTROLNET_germany*.json`: Germany dataset variants
- `config_CONTROLNET_hungary*.json`: Hungary dataset variants

**Key Parameters**:
```json
{
  "vae_def": {
    "controlnet_def": {
      "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
      "conditioning_embedding_in_channels": 8,
      "conditioning_embedding_num_channels": [8, 32, 64]
    }
  }
}
```

---

### configs/config_INFERENCE_*.json

Inference-specific configurations

**Variants**:
- `config_INFERENCE_v1.json` through `v4.json`
- `config_INFERENCE_norm_v1.json`: Normalized inference

---

## Jupyter Notebooks

### Generation.ipynb
Basic image generation exploration

### Generation2.ipynb
Advanced generation experiments

### Generation2_latent.ipynb
Latent space manipulation and generation

### VAE_playground.ipynb
VAE training and reconstruction experiments

### calc_fid.ipynb
Calculate FID (Fréchet Inception Distance) metrics

---

## Utility Files

### compute_oct_metrics.py
**Purpose**: Compute OCT-specific quality metrics

**Metrics**:
- SSIM (Structural Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- Custom OCT metrics

**Usage**:
```bash
python compute_oct_metrics.py --real /path/to/real --synthetic /path/to/synthetic
```

---

### requirements.txt

Key dependencies:
```
torch>=2.0.0
monai>=1.3.0
numpy
matplotlib
pillow
tqdm
wandb
```

---

### environment.yml

Conda environment specification for reproducibility

---

## Output Directories

### runs/

Training outputs organized by job name:
```
runs/
├── vae/
│   ├── model_best.pt
│   ├── model_epoch_*.pt
│   ├── config.json
│   ├── train_log.txt
│   └── reconstructions/
├── diffusion/
│   ├── diff_unet_ckpt.pt
│   ├── samples_epoch_*/
│   └── ...
└── controlnet_*/
    └── ...
```

---

### wandb/

Weights & Biases experiment tracking:
```
wandb/
├── run-{timestamp}-{id}/
│   ├── files/
│   │   ├── wandb-summary.json
│   │   ├── config.yaml
│   │   └── media/images/
│   └── logs/
└── latest-run -> run-{timestamp}-{id}/
```

---

## Key Workflows

### Full Training Pipeline

```bash
# Step 1: Train VAE
python train_vae.py --config ./configs/config_VAE.json

# Step 2: Encode images to latents (optional but recommended)
python scripts/encode_to_latents.py \
  --input_dir /path/to/train/images \
  --output_dir /path/to/latents/train \
  --autoencoder_path ./runs/vae/model_best.pt

# Step 3: Train diffusion model
python train_diffusion.py

# Step 4: Train ControlNet (optional)
python train_controlnet_modality.py

# Step 5: Generate images
python inference_controlnet.py --config ./configs/config_CONTROLNET_v1.json
```

### Inference Only

```bash
# Load pre-trained models and generate
python inference_controlnet.py --config ./configs/config_CONTROLNET_germany.json
```

---

## Important Notes

### Data Paths
All data paths in configs are absolute. Update these in configuration files:
- `image_dir`: Training images
- `latents_path`: Pre-encoded latents
- `trained_autoencoder_path`: VAE checkpoint
- `trained_unet_path`: Diffusion model checkpoint

### GPU Memory Management
- Use `num_splits` in VAE config for large images
- Enable `use_checkpointing` for gradient checkpointing
- Set `cache_rate=0` if RAM is limited

### Reproducibility
- Set seeds using `set_random_seeds()`
- Save configs with checkpoints
- Log hyperparameters to W&B

---

## Next Steps

- See [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md) for detailed training instructions
- See [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md) for generation workflows
- See [01_ARCHITECTURE.md](01_ARCHITECTURE.md) for architectural details
