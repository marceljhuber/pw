# MAISI Project Overview

## What is MAISI?

**MAISI** (Medical AI for Synthetic Imaging) is a state-of-the-art framework for generating high-resolution 3D CT medical images using deep learning. Developed by NVIDIA and NIH researchers, it addresses critical challenges in medical imaging:

- **Data scarcity**: Limited medical imaging datasets due to privacy concerns and acquisition costs
- **High annotation costs**: Expensive expert annotations required for training
- **Privacy concerns**: Need for synthetic data that maintains clinical utility without exposing patient information

## Key Capabilities

### 1. High-Resolution 3D CT Generation
- Generates realistic CT volumes up to **512 × 512 × 768 voxels**
- Flexible volume dimensions and voxel spacing
- Supports multiple anatomical regions (chest, abdomen, head, pelvis)

### 2. Conditional Generation with ControlNet
- **127 anatomical structures** can be used as conditioning
- Generate images with specific:
  - Organ segmentation masks
  - Tumor locations and types
  - Body regions (head-neck, chest, abdomen, pelvis)
  - Voxel spacing configurations

### 3. Foundation Model Approach
- Trained on **10,277 CT volumes** from 24 diverse public datasets
- Generalizable across different:
  - Body regions
  - Disease conditions
  - Imaging protocols
  - Patient demographics

## Three-Stage Architecture

```
Stage 1: VAE-GAN Volume Compression
├── Input: 3D CT images (H×W×D)
├── Encoder: Compresses to latent space
├── Decoder: Reconstructs images
└── Output: Latent features (reduced memory footprint)

Stage 2: Latent Diffusion Model
├── Input: Latent features + conditions
├── Noise Scheduler: Gradual denoising process
├── U-Net: Predicts noise at each timestep
└── Output: Generated latent features

Stage 3: ControlNet (Optional)
├── Input: Task-specific conditions (segmentation masks, tumor labels)
├── ControlNet: Modulates diffusion process
├── Frozen Diffusion Model: Preserves learned knowledge
└── Output: Condition-guided synthetic images
```

## Technical Innovations

### Tensor Splitting Parallelism (TSP)
- Novel technique for handling memory constraints in 3D generation
- Splits feature maps across multiple GPUs
- First framework to generate realistic 3D CT images larger than 512³ voxels

### Flexible Conditioning Mechanism
- Body region conditioning (head-neck, chest, abdomen, pelvis)
- Voxel spacing control (physical dimensions)
- Segmentation mask conditioning (127 anatomical structures)
- Tumor mask conditioning (5 tumor types)

## Real-World Applications

1. **Data Augmentation**: Improve downstream task performance
   - Demonstrated 4-7% improvement in tumor segmentation
   - Liver, lung, pancreas tumor detection enhanced

2. **Privacy-Preserving Sharing**: Generate synthetic datasets
   - Maintain clinical utility without patient data
   - Enable research collaboration

3. **Rare Disease Research**: Synthesize underrepresented conditions
   - Generate specific tumor types
   - Create diverse pathological scenarios

4. **Model Training**: Pre-training and transfer learning
   - Foundation models for medical imaging
   - Reduce annotation requirements

## This Repository Implementation

This repository contains a **reimplementation and extension** of MAISI for **OCT (Optical Coherence Tomography)** retinal imaging, adapting the framework for 2D medical images:

### Key Differences from Original MAISI
- **Modality**: OCT retinal scans instead of CT volumes
- **Dimensionality**: Primarily 2D with spatial dimensions instead of full 3D
- **Datasets**: KermanyV3 retinal imaging dataset
- **Conditional Tasks**:
  - Modality conditioning (different OCT imaging modalities)
  - Retinal pathology conditioning (RETOUCH dataset)

### Repository Structure
- `train_vae.py`: Train the VAE-GAN volume compression network
- `train_diffusion.py`: Train the latent diffusion model
- `train_controlnet_modality.py`: Train ControlNet for modality conditioning
- `train_controlnet_retouch.py`: Train ControlNet for retinal pathology
- `inference*.py`: Various inference scripts for generation
- `networks/`: Model architectures (AutoencoderKlMaisi, ControlNetMaisi)
- `scripts/`: Utility functions for training, inference, and data processing

## Quick Start

### Training Pipeline
```bash
# 1. Train VAE-GAN
python train_vae.py --config ./configs/config_VAE.json

# 2. Encode images to latents
python scripts/encode_to_latents.py \
  --input_dir /path/to/images \
  --output_dir /path/to/latents \
  --autoencoder_path ./runs/vae_run/model_best.pt

# 3. Train Diffusion Model
python train_diffusion.py

# 4. (Optional) Train ControlNet
python train_controlnet_modality.py  # or train_controlnet_retouch.py
```

### Inference
```bash
python inference_controlnet.py --config ./configs/config_CONTROLNET_v1.json
```

## Performance Metrics

Based on the MAISI paper results:
- **FID Score**: 3.301 (vs 98.208 for HA-GAN baseline) on autoPET 2023 dataset
- **Data Augmentation**: 4-7% improvement in Dice Similarity Coefficient for tumor segmentation
- **Image Quality**: Realistic anatomical structures with high fidelity

## References

- **Paper**: "MAISI: Medical AI for Synthetic Imaging" (arXiv:2409.11169v3)
- **Authors**: Pengfei Guo, Can Zhao, Dong Yang, et al. (NVIDIA, NIH)
- **Code**: Based on MONAI framework
- **GitHub**: NVIDIA MedTech repository

## Next Steps

For detailed information, see:
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) - Technical architecture details
- [02_PAPER_SUMMARY.md](02_PAPER_SUMMARY.md) - Comprehensive paper summary
- [03_CODEBASE_MAP.md](03_CODEBASE_MAP.md) - Code organization guide
- [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md) - Training instructions
- [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md) - Generation instructions
