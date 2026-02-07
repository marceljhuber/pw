# MAISI Inference Guide

Complete guide for generating synthetic medical images using trained MAISI models.

## Overview

MAISI supports three types of inference:
1. **Unconditional Generation**: Random sampling from learned distribution
2. **Conditional Generation**: Generate with primary conditions (body region, spacing)
3. **ControlNet Generation**: Fine-grained control with task-specific inputs

---

## Prerequisites

### Trained Models

You need at least:
- **VAE checkpoint**: `runs/vae/model_best.pt`
- **Diffusion model checkpoint**: `runs/diffusion/model_best.pt`
- **(Optional) ControlNet checkpoint**: `runs/controlnet/model_best.pt`

### Environment

```bash
# Activate environment
conda activate maisi

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Basic Inference: Diffusion Model

### Configuration

Create/edit `configs/config_INFERENCE_v1.json`:

```json
{
  "main": {
    "trained_autoencoder_path": "./runs/vae_oct/model_best.pt",
    "trained_unet_path": "./runs/diffusion/model_best.pt",
    "output_dir": "./outputs/generated",
    "num_samples": 10,
    "random_seed": 42
  },
  "inference": {
    "num_inference_steps": 1000,    // DDPM: 1000, DDIM: 50-100
    "guidance_scale": 1.0,           // CFG scale (1.0 = no guidance)
    "scheduler": "ddpm"              // "ddpm" or "ddim"
  },
  "model_config": {
    "diffusion_unet_inference": {
      "dim": [256, 256, 128],        // Output dimensions
      "spacing": [1.0, 1.0, 1.0],   // Voxel spacing (mm)
      "top_region_index": [0, 1, 0, 0],    // Body region: chest
      "bottom_region_index": [0, 0, 1, 0]  // Body region: abdomen
    }
  }
}
```

### Inference Command

```bash
python inference.py --config ./configs/config_INFERENCE_v1.json
```

### Expected Output

```
Loading VAE from: ./runs/vae_oct/model_best.pt
Loading Diffusion Model from: ./runs/diffusion/model_best.pt
Initializing DDPM Scheduler (1000 steps)

Generating sample 1/10...
Denoising: 100%|████████| 1000/1000 [02:34<00:00, 6.47it/s]
Saved: ./outputs/generated/sample_001.png

Generating sample 2/10...
...

Generation complete! 10 samples saved to ./outputs/generated/
```

---

## Fast Inference: DDIM

DDIM enables much faster sampling with fewer steps.

### Configuration Changes

```json
{
  "inference": {
    "num_inference_steps": 50,      // Much fewer steps
    "scheduler": "ddim",
    "eta": 0.0                       // 0.0 = deterministic, 1.0 = stochastic
  }
}
```

### Speed Comparison

| Scheduler | Steps | Time per Sample | Quality |
|-----------|-------|-----------------|---------|
| DDPM      | 1000  | ~2.5 min        | Best    |
| DDIM      | 100   | ~15 sec         | Very Good |
| DDIM      | 50    | ~8 sec          | Good    |
| DDIM      | 25    | ~4 sec          | Acceptable |

### Usage

```bash
python inference_optimized.py --config ./configs/config_INFERENCE_v1.json
```

---

## Conditional Generation

### Body Region Conditioning

Generate images for specific body regions:

```json
{
  "model_config": {
    "diffusion_unet_inference": {
      // Head-Neck region
      "top_region_index": [1, 0, 0, 0],
      "bottom_region_index": [1, 0, 0, 0],

      // OR Chest region
      "top_region_index": [0, 1, 0, 0],
      "bottom_region_index": [0, 1, 0, 0],

      // OR Abdomen region
      "top_region_index": [0, 0, 1, 0],
      "bottom_region_index": [0, 0, 1, 0],

      // OR Pelvis region
      "top_region_index": [0, 0, 0, 1],
      "bottom_region_index": [0, 0, 0, 1],

      // OR Multi-region (chest to abdomen)
      "top_region_index": [0, 1, 0, 0],
      "bottom_region_index": [0, 0, 1, 0]
    }
  }
}
```

### Voxel Spacing Conditioning

Control physical dimensions:

```json
{
  "model_config": {
    "diffusion_unet_inference": {
      // High resolution (0.5mm voxels)
      "spacing": [0.5, 0.5, 0.5],

      // OR Standard resolution (1.0mm voxels)
      "spacing": [1.0, 1.0, 1.0],

      // OR Low resolution (2.0mm voxels)
      "spacing": [2.0, 2.0, 2.0],

      // OR Anisotropic (different spacing per axis)
      "spacing": [0.5, 0.5, 1.0]
    }
  }
}
```

### Volume Dimension Conditioning

Control output size:

```json
{
  "model_config": {
    "diffusion_unet_inference": {
      // Small volume
      "dim": [128, 128, 64],

      // OR Medium volume
      "dim": [256, 256, 128],

      // OR Large volume
      "dim": [512, 512, 256],

      // OR Very large (requires TSP)
      "dim": [512, 512, 768]
    }
  }
}
```

---

## Classifier-Free Guidance (CFG)

CFG improves conditioning adherence at the cost of diversity.

### Configuration

```json
{
  "inference": {
    "guidance_scale": 7.5,    // Typical range: 1.0 to 15.0
    "use_cfg": true
  }
}
```

### Guidance Scale Effects

| Scale | Effect | Use Case |
|-------|--------|----------|
| 1.0   | No guidance, maximum diversity | Exploration, unconditional |
| 3.0   | Light guidance, diverse | Slight conditioning |
| 7.5   | Moderate guidance, balanced | **Recommended default** |
| 10.0  | Strong guidance, less diverse | Specific requirements |
| 15.0  | Very strong, may sacrifice quality | Extreme conditioning |

### Example

```python
# Generate samples with varying guidance scales
for scale in [1.0, 3.0, 7.5, 10.0]:
    config['inference']['guidance_scale'] = scale
    generate_samples(config)
```

---

## ControlNet Inference

### Segmentation Mask Conditioning

Generate CT images matching anatomical segmentation masks.

#### Configuration

```json
{
  "main": {
    "trained_autoencoder_path": "./runs/vae_oct/model_best.pt",
    "trained_unet_path": "./runs/diffusion/model_best.pt",
    "trained_controlnet_path": "./runs/controlnet/model_best.pt",
    "conditioning_input": "./data/masks/liver_tumor_mask.png",
    "output_dir": "./outputs/controlnet_generated"
  },
  "inference": {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "controlnet_scale": 1.0      // Control strength (0.0 to 2.0)
  }
}
```

#### Inference Command

```bash
python inference_controlnet.py --config ./configs/config_CONTROLNET_v1.json
```

#### Control Scale Effects

| Scale | Effect |
|-------|--------|
| 0.0   | No control (same as base diffusion) |
| 0.5   | Light control, more variation |
| 1.0   | **Default**, balanced control |
| 1.5   | Strong control, adheres closely |
| 2.0   | Very strong, may sacrifice realism |

### Modality Conditioning

Generate images in specific OCT modalities.

#### Configuration

```json
{
  "main": {
    "trained_controlnet_path": "./runs/controlnet_modality/model_best.pt",
    "modality": "spectralis",     // or "cirrus", "topcon", etc.
    "modality_id": 0              // Numeric encoding
  }
}
```

#### Usage

```bash
python inference_controlnet.py --config ./configs/config_CONTROLNET_modality.json
```

### Tumor Inpainting

Generate realistic tumors in existing CT images.

#### Workflow

```python
# 1. Load real patient CT
real_ct = load_ct("patient_001.nii.gz")

# 2. Generate or load tumor mask
tumor_mask = generate_tumor_mask(
    location=[128, 128, 64],
    size=[20, 20, 15],
    tumor_type="liver"
)

# 3. Run ControlNet inpainting
synthetic_ct = controlnet_inpaint(
    real_ct,
    tumor_mask,
    tumor_type="liver"
)

# 4. Save result
save_ct(synthetic_ct, "patient_001_with_tumor.nii.gz")
```

---

## Batch Generation

### Generate Multiple Samples

```bash
# Generate 100 samples
python inference.py \
  --config ./configs/config_INFERENCE_v1.json \
  --num_samples 100 \
  --output_dir ./outputs/batch_generation
```

### Programmatic Batch Generation

```python
import torch
from inference import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    vae_path="./runs/vae/model_best.pt",
    unet_path="./runs/diffusion/model_best.pt"
)

# Generate batch
samples = []
for i in range(100):
    # Random seed for diversity
    torch.manual_seed(42 + i)

    # Generate sample
    sample = pipeline.generate(
        num_steps=50,
        guidance_scale=7.5,
        body_region="chest"
    )

    samples.append(sample)
    print(f"Generated {i+1}/100")

# Save samples
save_batch(samples, "./outputs/batch/")
```

---

## Advanced Techniques

### Latent Space Interpolation

Smoothly interpolate between two generated images.

```python
# Generate two samples
z1 = torch.randn(1, 4, 32, 32)
z2 = torch.randn(1, 4, 32, 32)

# Interpolate in latent space
alphas = torch.linspace(0, 1, steps=10)
interpolated = []

for alpha in alphas:
    z_interp = (1 - alpha) * z1 + alpha * z2

    # Denoise from interpolated latent
    sample = diffusion_model.sample_from_latent(z_interp)

    interpolated.append(sample)

# Save interpolation sequence
save_sequence(interpolated, "interpolation.gif")
```

### Latent Space Manipulation

Edit specific attributes by manipulating latents.

```python
# Generate base sample
z = torch.randn(1, 4, 32, 32)
sample = generate(z)

# Find direction in latent space (e.g., brightness)
direction = compute_attribute_direction("brightness")

# Manipulate latent
z_bright = z + 2.0 * direction
z_dark = z - 2.0 * direction

sample_bright = generate(z_bright)
sample_dark = generate(z_dark)
```

### Conditional Interpolation

Smoothly transition between conditions.

```python
# Start: chest region
condition_start = {"body_region": [0, 1, 0, 0]}

# End: abdomen region
condition_end = {"body_region": [0, 0, 1, 0]}

# Interpolate conditions
for alpha in torch.linspace(0, 1, 10):
    condition = {
        "body_region": (1-alpha) * condition_start["body_region"]
                       + alpha * condition_end["body_region"]
    }

    sample = generate(condition=condition)
    save(sample, f"transition_{alpha:.1f}.png")
```

---

## Quality Control

### Visual Inspection

```python
import matplotlib.pyplot as plt

def inspect_samples(samples, num_display=16):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
        ax.imshow(sample, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {idx+1}')

    plt.tight_layout()
    plt.savefig('sample_inspection.png', dpi=150)
    plt.show()
```

### Quantitative Metrics

```python
from scripts.utils import compute_metrics

# Generate samples
samples = [generate() for _ in range(100)]

# Load real images for comparison
real_images = load_real_images()

# Compute metrics
metrics = compute_metrics(samples, real_images)
print(f"FID: {metrics['fid']:.2f}")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

### Anatomical Consistency Check

```python
def check_anatomical_consistency(sample):
    """
    Verify generated images have plausible anatomy.
    """
    # Run segmentation model
    segmentation = segment(sample)

    # Check organ presence
    organs_present = detect_organs(segmentation)

    # Verify organ sizes are realistic
    organ_volumes = compute_volumes(segmentation)

    # Check spatial relationships
    relationships_valid = verify_relationships(segmentation)

    return {
        'organs_present': organs_present,
        'organ_volumes': organ_volumes,
        'relationships_valid': relationships_valid
    }
```

---

## Inference Optimization

### GPU Memory Optimization

```python
# Use half precision
model.half()

# Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True

# Clear cache between batches
torch.cuda.empty_cache()
```

### Multi-GPU Inference

```python
# Distribute batch across GPUs
from torch.nn import DataParallel

model = DataParallel(model, device_ids=[0, 1, 2, 3])

# Generate in parallel
batch_size = 16  # 4 per GPU
samples = model(noise_batch)
```

### CPU Inference (Slow)

```python
# For systems without GPU
device = torch.device("cpu")
model = model.to(device)

# Use DDIM with minimal steps
num_steps = 25  # Minimum for acceptable quality
```

---

## Inference Scripts Reference

### inference.py

**Purpose**: Basic diffusion model inference

**Key Parameters**:
- `--config`: Path to config file
- `--num_samples`: Number of samples to generate
- `--output_dir`: Output directory
- `--seed`: Random seed

**Example**:
```bash
python inference.py \
  --config ./configs/config_INFERENCE_v1.json \
  --num_samples 50 \
  --seed 123
```

---

### inference_controlnet.py

**Purpose**: ControlNet conditional generation

**Key Parameters**:
- `--config`: Path to ControlNet config
- `--conditioning_input`: Path to conditioning input (mask, image, etc.)
- `--control_scale`: ControlNet influence (0.0-2.0)

**Example**:
```bash
python inference_controlnet.py \
  --config ./configs/config_CONTROLNET_germany.json \
  --conditioning_input ./data/masks/tumor_mask.png \
  --control_scale 1.5
```

---

### inference_optimized.py

**Purpose**: Fast inference with DDIM

**Features**:
- DDIM sampling (50 steps vs 1000 for DDPM)
- Memory-optimized
- Batch processing support

**Example**:
```bash
python inference_optimized.py \
  --num_steps 50 \
  --batch_size 8 \
  --output_dir ./outputs/fast_generation
```

---

## Troubleshooting

### Poor Sample Quality

**Symptom**: Blurry, unrealistic images

**Solutions**:
1. Increase inference steps (50 → 100 for DDIM)
2. Use higher guidance scale (7.5 → 10.0)
3. Check VAE quality first
4. Verify diffusion model trained sufficiently

---

### Samples Don't Match Conditions

**Symptom**: Generated images ignore body region / ControlNet input

**Solutions**:
1. Increase guidance scale
2. Increase ControlNet scale (1.0 → 1.5)
3. Verify conditioning input is preprocessed correctly
4. Check model was trained with conditional inputs

---

### Out of Memory During Inference

**Symptom**: CUDA out of memory error

**Solutions**:
1. Reduce batch size (16 → 8 → 4 → 1)
2. Use DDIM with fewer steps
3. Generate smaller dimensions
4. Use half precision (`.half()`)
5. Clear cache: `torch.cuda.empty_cache()`

---

### Slow Generation Speed

**Symptom**: Takes too long per sample

**Solutions**:
1. Switch from DDPM (1000 steps) to DDIM (50 steps)
2. Use half precision
3. Enable CUDA benchmarking
4. Use multiple GPUs for batch
5. Reduce output dimensions

---

## Output Formats

### Save as PNG

```python
from PIL import Image
import numpy as np

# Convert tensor to numpy
sample_np = sample.cpu().numpy()

# Normalize to [0, 255]
sample_np = (sample_np * 255).astype(np.uint8)

# Save
Image.fromarray(sample_np).save("sample.png")
```

### Save as NIfTI (Medical Format)

```python
import nibabel as nib

# Convert tensor to numpy
volume_np = volume.cpu().numpy()

# Create NIfTI image
nifti_img = nib.Nifti1Image(volume_np, affine=np.eye(4))

# Set spacing
nifti_img.header.set_zooms([1.0, 1.0, 1.0])

# Save
nib.save(nifti_img, "generated_volume.nii.gz")
```

### Save as DICOM

```python
import pydicom
from pydicom.dataset import Dataset

# Create DICOM dataset
ds = Dataset()
ds.PatientName = "Generated^Patient"
ds.Modality = "CT"
ds.PixelData = sample_np.tobytes()
ds.Rows, ds.Columns = sample_np.shape

# Save
ds.save_as("generated.dcm")
```

---

## Example Workflows

### Workflow 1: Generate Dataset for Training

```bash
# Generate 1000 synthetic images
python inference.py \
  --config ./configs/config_INFERENCE_v1.json \
  --num_samples 1000 \
  --output_dir ./synthetic_dataset/train \
  --seed 42

# Generate 200 for validation
python inference.py \
  --config ./configs/config_INFERENCE_v1.json \
  --num_samples 200 \
  --output_dir ./synthetic_dataset/val \
  --seed 999
```

### Workflow 2: Generate Tumor Dataset

```bash
# Generate diverse tumors
for tumor_type in liver pancreas lung colon bone; do
  python inference_controlnet.py \
    --config ./configs/config_CONTROLNET_tumor.json \
    --tumor_type $tumor_type \
    --num_samples 100 \
    --output_dir ./synthetic_tumors/$tumor_type
done
```

### Workflow 3: Multi-Condition Sweep

```python
# Generate images across all conditions
body_regions = ['head', 'chest', 'abdomen', 'pelvis']
spacings = [0.5, 1.0, 1.5, 2.0]

for region in body_regions:
    for spacing in spacings:
        config = load_config()
        config['body_region'] = region
        config['spacing'] = [spacing, spacing, spacing]

        generate_batch(
            config=config,
            num_samples=25,
            output_dir=f"./outputs/{region}_spacing{spacing}"
        )
```

---

## Next Steps

- Experiment with different conditioning strategies
- Evaluate synthetic data on downstream tasks
- Fine-tune generation parameters for your specific use case
- Integrate generated data into training pipelines

For more details:
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) - Model architecture
- [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md) - Training new models
- [03_CODEBASE_MAP.md](03_CODEBASE_MAP.md) - Code organization
