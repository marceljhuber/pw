# MAISI Technical Architecture

## Overview

MAISI employs a three-stage hierarchical architecture combining volume compression, latent diffusion, and conditional control mechanisms.

## Stage 1: Volume Compression Network (VAE-GAN)

### Architecture: AutoencoderKlMaisi

**Purpose**: Compress high-resolution 3D medical images into compact latent representations

```
Input Image: x ∈ ℝ^(H×W×D)
    ↓
Encoder ℰ: Spatial + Channel Downsampling
    ↓
Latent Space: z = ℰ(x) ∈ ℝ^(H'×W'×D'×C)
    ↓
Decoder 𝒟: Spatial + Channel Upsampling
    ↓
Reconstructed: x̂ = 𝒟(ℰ(x))
```

### Key Components

#### 1. **Encoder Architecture**
- **Input**: 1-channel grayscale medical images
- **Downsampling Stages**: 3 stages with [64, 128, 256] channels
- **Residual Blocks**: 2 blocks per stage for feature extraction
- **Output**: 4-channel latent representation (compression ratio ~8x)

#### 2. **Decoder Architecture**
- **Input**: 4-channel latent features
- **Upsampling Stages**: 3 stages mirroring encoder
- **Transpose Convolutions**: Optional (controlled by `use_convtranspose`)
- **Output**: Reconstructed 1-channel image

#### 3. **Normalization: MaisiGroupNorm3D**
- Custom Group Normalization for 3D medical images
- **Float16 optimization**: Optional conversion to FP16 (`norm_float16=True`)
- **Memory saving**: CUDA cache clearing after operations
- **Groups**: 32 groups for stable training

#### 4. **Tensor Splitting Parallelism (TSP)**
- **Purpose**: Handle large 3D volumes that exceed GPU memory
- **Method**: Split feature maps along spatial dimensions
- **Parameters**:
  - `num_splits`: Number of splits (e.g., 8 for large volumes)
  - `dim_split`: Dimension to split along (1=height, 2=width, 3=depth)
- **Process**:
  ```
  1. Partition input into overlapping segments
  2. Process each segment on separate devices/sequentially
  3. Stitch outputs using normalization layers
  ```

### Loss Functions

The VAE-GAN is trained with a combination of losses:

```python
ℒ_AE = min/max_ℰ,𝒟,Disc (
    ℒ_recon(x, 𝒟(ℰ(x))) +           # Reconstruction (L1/L2)
    ℒ_lpips(x, 𝒟(ℰ(x))) +           # Perceptual loss
    ℒ_adv(ℰ, 𝒟) +                    # Adversarial loss
    ℒ_reg(ℰ(x)) +                    # KL regularization
    ℒ_adv
)
```

Where:
- **ℒ_recon**: Pixel-wise reconstruction loss (L1 or MSE)
  - Weight: 1.0 (default)
- **ℒ_lpips**: Perceptual loss using pre-trained network
  - Weight: 0.1
  - Uses SqueezeNet for feature extraction
- **ℒ_adv**: GAN adversarial loss
  - Weight: 0.05
  - PatchGAN discriminator with 3 layers
- **ℒ_KL**: KL divergence regularization
  - Weight: 1e-6
  - Standard deviation clamped to [0.9, 1.1]

### Configuration Example

```json
{
  "model": {
    "autoencoder": {
      "spatial_dims": 2,              // 2D or 3D
      "in_channels": 1,               // Grayscale input
      "out_channels": 1,              // Grayscale output
      "latent_channels": 4,           // Latent representation
      "num_channels": [64, 128, 256], // Channel progression
      "num_res_blocks": [2, 2, 2],    // Residual blocks per stage
      "norm_num_groups": 32,          // Group norm groups
      "attention_levels": [false, false, false], // Attention per stage
      "norm_float16": true,           // FP16 normalization
      "num_splits": 8,                // TSP splits
      "dim_split": 1                  // Split dimension
    }
  }
}
```

## Stage 2: Latent Diffusion Model

### Architecture: DiffusionModelUNetMaisi

**Purpose**: Generate realistic latent features through iterative denoising

### Diffusion Process

#### Forward (Noise Addition)
```
zt = √ᾱt · z0 + √(1-ᾱt) · ε,  ε ~ 𝒩(0, I)

where:
- z0: Clean latent features
- zt: Noisy latent at timestep t
- ᾱt: Noise schedule coefficient
- ε: Gaussian noise
```

#### Reverse (Denoising)
```
Training objective:
ℒθ(t) = 𝔼t,z0,ε [‖ε - εθ(zt, t, cp, cT)‖₁]

where:
- εθ: U-Net noise predictor
- cp: Primary conditioning (body region, voxel spacing)
- cT: Task-specific conditioning (from ControlNet)
```

### U-Net Architecture

```
Time Embedding (t)
    ↓
┌─────────────────────────────────────┐
│ Downsampling Path                   │
├─────────────────────────────────────┤
│ [64] → ResBlock × 2                 │
│ [128] → ResBlock × 2                │
│ [256] → ResBlock × 2 + Attention    │
│ [512] → ResBlock × 2 + Attention    │
└─────────────────────────────────────┘
           ↓
    Bottleneck
           ↓
┌─────────────────────────────────────┐
│ Upsampling Path (Skip Connections)  │
├─────────────────────────────────────┤
│ [512] → ResBlock × 2 + Attention    │
│ [256] → ResBlock × 2 + Attention    │
│ [128] → ResBlock × 2                │
│ [64] → ResBlock × 2                 │
└─────────────────────────────────────┘
           ↓
    Predicted Noise ε̂
```

### Conditioning Mechanisms

#### Primary Conditioning (Built-in)
```python
# Body Region Encoding
itop = [1, 0, 0, 0]    # Head-neck
ichest = [0, 1, 0, 0]  # Chest
iabdomen = [0, 0, 1, 0] # Abdomen
ipelvic = [0, 0, 0, 1]  # Pelvis

# Voxel Spacing
s = [sx, sy, sz]  # mm per voxel
```

#### Task-Specific Conditioning (ControlNet)
- Segmentation masks (127 anatomical structures)
- Tumor masks (5 tumor types)
- Modality-specific features
- Custom conditioning inputs

### Noise Schedulers

#### 1. **DDPM (Denoising Diffusion Probabilistic Models)**
```python
"noise_scheduler": {
  "num_train_timesteps": 1000,
  "beta_start": 0.0015,
  "beta_end": 0.0195,
  "schedule": "scaled_linear_beta",
  "clip_sample": false
}
```

#### 2. **DDIM (Denoising Diffusion Implicit Models)**
- Faster sampling with fewer steps
- Deterministic generation (when η=0)
- Configuration in `ddim.py` and `ddim_hacked.py`

### Training Configuration

```json
{
  "diffusion_unet_train": {
    "batch_size": 8,
    "lr": 0.0001,
    "n_epochs": 1000,
    "cache_rate": 0  // Data caching for speed
  }
}
```

## Stage 3: ControlNet

### Architecture: ControlNetMaisi

**Purpose**: Provide fine-grained control over generation without retraining the diffusion model

### Design Philosophy

```
┌──────────────────────────────┐
│ Frozen Diffusion Model       │  ← Preserves learned knowledge
│ (Locked weights)             │
└──────────────────────────────┘
            ↑
      Zero Convolutions
            ↑
┌──────────────────────────────┐
│ Trainable ControlNet Copy    │  ← Learns task-specific control
│ (Same architecture as U-Net) │
└──────────────────────────────┘
            ↑
    Condition Encoder
            ↑
┌──────────────────────────────┐
│ Task-Specific Input          │
│ (Masks, Images, etc.)        │
└──────────────────────────────┘
```

### ControlNet Components

#### 1. **Conditioning Encoder**
```python
"conditioning_embedding_in_channels": 8,    // Input channels
"conditioning_embedding_num_channels": [8, 32, 64]  // Encoder progression
```

Transforms conditioning input (e.g., segmentation mask) into latent features compatible with U-Net.

#### 2. **ControlNet Backbone**
- **Architecture**: Identical to diffusion U-Net
- **Initialization**: Copy of pre-trained U-Net weights
- **Training**: Weights evolve from pre-trained to task-specific

#### 3. **Zero Convolutions**
- **Purpose**: Ensure ControlNet starts with zero influence
- **Method**: Convolution layers initialized to output zeros
- **Benefit**: Gradual learning without disrupting pre-trained model

### Integration with Diffusion Model

```python
# Training objective
ℒControlNet(t) = 𝔼[‖ε - εθ(zt, t, cp, cT)‖₁]

where:
- cT = ControlNet(condition_input)
- Diffusion model parameters θ are frozen
- Only ControlNet parameters are updated
```

### Supported Conditioning Tasks

#### 1. **MAISI CT Generation**
- 127 anatomical structures (TotalSegmentator)
- Body region specification
- Voxel spacing control

#### 2. **MAISI Tumor Inpainting**
- 5 tumor types: liver, pancreas, lung, colon, bone lesion
- Synthetic tumor mask generation
- Realistic tumor integration

#### 3. **Modality Conditioning** (This Repo)
- Different OCT imaging modalities
- Scanner-specific characteristics

#### 4. **RETOUCH Pathology** (This Repo)
- Retinal fluid segmentation
- Disease-specific features

### Training Strategy

```python
# Phase 1: Train diffusion model on unlabeled data
train_diffusion_model(latents_only=True)

# Phase 2: Train ControlNet with labeled data
# - Freeze diffusion model weights
# - Train only ControlNet + zero conv layers
train_controlnet(
    frozen_diffusion=True,
    conditioning_data=labeled_masks
)
```

## Memory Optimization Techniques

### 1. Tensor Splitting Parallelism (TSP)

```python
# Example: Split 512×512×768 volume across 8 segments
feature_maps = split_tensor(
    input=volume,
    num_splits=8,
    dim=1,  # Split along height
    overlap=16  # Overlap for continuity
)

outputs = []
for segment in feature_maps:
    output = process_segment(segment)
    outputs.append(output)

result = stitch_with_normalization(outputs)
```

### 2. Gradient Checkpointing
```python
"use_checkpointing": true  // Trade compute for memory
```

### 3. Mixed Precision Training (AMP)
```python
"amp": true  // FP16 for forward/backward, FP32 for optimizer
```

### 4. Normalization Float16
```python
"norm_float16": true  // FP16 group norm outputs
```

## Inference Pipeline

### Standard Generation
```
1. Sample noise: z_T ~ 𝒩(0, I)
2. Set conditions: body_region, voxel_spacing
3. For t = T to 1:
     z_{t-1} = denoise_step(z_t, t, conditions)
4. Decode: x̂ = Decoder(z_0)
```

### ControlNet Generation
```
1. Sample noise: z_T ~ 𝒩(0, I)
2. Set conditions: body_region, voxel_spacing, task_input
3. Encode condition: c_T = ControlNet.encode(task_input)
4. For t = T to 1:
     control = ControlNet(z_t, t, c_T)
     z_{t-1} = denoise_step(z_t, t, control)
5. Decode: x̂ = Decoder(z_0)
```

### Classifier-Free Guidance (CFG)

Improves condition adherence:
```python
ε̂ = ε_uncond + w · (ε_cond - ε_uncond)

where:
- w: Guidance scale (typical: 1.0-7.5)
- ε_cond: Noise prediction with conditioning
- ε_uncond: Noise prediction without conditioning
```

## Configuration Files Structure

### VAE Config (`config_VAE.json`)
- Model architecture: encoder/decoder specs
- Training hyperparameters: LR, batch size, epochs
- Loss weights: KL, perceptual, adversarial
- Data augmentation: transforms, normalization

### Diffusion Config (`config_DIFF.json`)
- U-Net architecture
- Noise scheduler parameters
- Conditioning setup: body regions, spacing
- Training configuration

### ControlNet Config (`config_CONTROLNET_*.json`)
- ControlNet architecture (matches U-Net)
- Conditioning encoder specs
- Task-specific settings (modality, RETOUCH, etc.)
- Dataset paths and preprocessing

## Next Steps

- See [02_PAPER_SUMMARY.md](02_PAPER_SUMMARY.md) for research context
- See [03_CODEBASE_MAP.md](03_CODEBASE_MAP.md) for implementation details
- See [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md) for training procedures
