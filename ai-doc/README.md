# AI Documentation for MAISI Project

This directory contains comprehensive documentation about the MAISI (Medical AI for Synthetic Imaging) project, specifically designed for AI agents to understand and work with the codebase.

## Purpose

This documentation serves as a knowledge base for:
- **AI Coding Assistants** (like Claude, GPT-4, etc.) to understand the project structure and implementation
- **Developers** joining the project to get up to speed quickly
- **Researchers** wanting to understand the MAISI framework and its adaptation for OCT imaging

## Quick Navigation

### 📖 Core Documentation

1. **[00_OVERVIEW.md](00_OVERVIEW.md)** - Start here!
   - High-level overview of MAISI
   - Key capabilities and innovations
   - Quick start guide
   - Project context

2. **[01_ARCHITECTURE.md](01_ARCHITECTURE.md)** - Technical deep dive
   - Three-stage architecture (VAE-GAN, Diffusion, ControlNet)
   - Mathematical formulations
   - Model components and configurations
   - Tensor Splitting Parallelism (TSP)

3. **[02_PAPER_SUMMARY.md](02_PAPER_SUMMARY.md)** - Research paper summary
   - Comprehensive summary of the MAISI paper (arXiv:2409.11169v3)
   - Methodology details
   - Experimental results and datasets
   - Comparison with baselines

4. **[03_CODEBASE_MAP.md](03_CODEBASE_MAP.md)** - Code organization
   - Repository structure
   - Key files and their purposes
   - Training and inference scripts
   - Configuration files
   - Network architectures

5. **[04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md)** - Training instructions
   - Step-by-step training procedures
   - Configuration setup
   - Hyperparameter tuning
   - Troubleshooting common issues
   - Hardware optimizations

6. **[05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md)** - Generation guide
   - Inference workflows
   - Conditional generation
   - ControlNet usage
   - Batch generation
   - Quality control

7. **[06_DATASET_KERMANYV3.md](06_DATASET_KERMANYV3.md)** - Dataset documentation
   - Dataset structure and statistics
   - Clinical context for each class
   - Data loading examples
   - Preprocessing recommendations
   - Class imbalance handling

8. **[07_FID_LEARNINGS.md](07_FID_LEARNINGS.md)** - FID debugging learnings
   - Root causes observed in this codebase
   - What was fixed already
   - Practical strategy to reduce FID
   - Evaluation protocol recommendations

9. **[08_VAE_IMPL_DIFF_AND_ACTION_PLAN.md](08_VAE_IMPL_DIFF_AND_ACTION_PLAN.md)** - VAE gap analysis
   - Differences vs original MAISI intent
   - Why current quality is low
   - 3 new 128px full-data VAE configs
   - Recommended run order to improve FID

## Project Overview

### What is MAISI?

MAISI (Medical AI for Synthetic Imaging) is a framework for generating high-resolution 3D medical CT images using:
- **Stage 1**: VAE-GAN for volume compression
- **Stage 2**: Latent Diffusion Model for generation
- **Stage 3**: ControlNet for conditional control

### This Implementation

This repository adapts MAISI for **OCT (Optical Coherence Tomography)** retinal imaging:
- Primarily 2D with spatial dimensions (vs full 3D CT)
- OCT-specific datasets (KermanyV3)
- Modality and pathology conditioning

### Key Features

- 🚀 **High-resolution generation**: Up to 512×512×768 voxels
- 🎯 **Flexible conditioning**: Body regions, voxel spacing, anatomical structures
- 🧠 **Foundation model**: Trained on diverse datasets for generalization
- ⚡ **Tensor Splitting Parallelism**: Handle large volumes on limited GPU memory
- 🎨 **ControlNet**: Fine-grained control without retraining

## Quick Start

### For AI Agents

If you're an AI agent helping with this codebase:

1. **Read [00_OVERVIEW.md](00_OVERVIEW.md)** first to understand the project
2. **Check [03_CODEBASE_MAP.md](03_CODEBASE_MAP.md)** to locate specific files
3. **Refer to [01_ARCHITECTURE.md](01_ARCHITECTURE.md)** for technical details
4. **Use [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md)** for training tasks
5. **Use [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md)** for generation tasks

### For Developers

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate maisi

# 2. Train VAE
python train_vae.py --config ./configs/config_VAE.json

# 3. Train Diffusion Model
python train_diffusion.py

# 4. Generate images
python inference.py --config ./configs/config_INFERENCE_v1.json
```

See [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md) for detailed instructions.

## Repository Structure

```
pw/
├── ai-doc/              # This directory - AI agent documentation
├── configs/             # Training and inference configurations
├── networks/            # Model architectures
│   ├── autoencoderkl_maisi.py
│   ├── controlnet_maisi.py
│   └── schedulers/
├── scripts/             # Utility functions
│   ├── diff_model_train.py
│   ├── train_controlnet.py
│   ├── utils.py
│   └── utils_data.py
├── runs/                # Training outputs and checkpoints
├── train_*.py           # Training entry points
├── inference*.py        # Inference scripts
└── requirements.txt     # Python dependencies
```

## Common Tasks

### Training a New Model

See [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md)

```bash
# Stage 1: VAE
python train_vae.py --config ./configs/config_VAE.json

# Stage 2: Diffusion
python train_diffusion.py

# Stage 3: ControlNet
python train_controlnet_modality.py
```

### Generating Images

See [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md)

```bash
# Basic generation
python inference.py --config ./configs/config_INFERENCE_v1.json

# ControlNet generation
python inference_controlnet.py --config ./configs/config_CONTROLNET_v1.json

# Fast generation (DDIM)
python inference_optimized.py
```

### Understanding Model Architecture

See [01_ARCHITECTURE.md](01_ARCHITECTURE.md)

Key components:
- **AutoencoderKlMaisi**: VAE with Tensor Splitting Parallelism
- **DiffusionModelUNetMaisi**: Latent diffusion U-Net
- **ControlNetMaisi**: Conditional control network
- **Noise Schedulers**: DDPM, DDIM

### Modifying Configurations

See [03_CODEBASE_MAP.md](03_CODEBASE_MAP.md) for config file details

Configuration files in `configs/`:
- `config_VAE.json`: VAE training
- `config_DIFF.json`: Diffusion training
- `config_CONTROLNET_*.json`: ControlNet variants
- `config_INFERENCE_*.json`: Inference settings

## Key Concepts

### Tensor Splitting Parallelism (TSP)

Novel technique for handling large 3D volumes:
- Splits feature maps across spatial dimensions
- Processes segments independently or across GPUs
- Stitches results using normalization layers
- Enables generation of 512³+ volumes on consumer GPUs

See [01_ARCHITECTURE.md](01_ARCHITECTURE.md#tensor-splitting-parallelism-tsp) for details.

### Latent Diffusion

Generate in compressed latent space instead of pixel space:
- VAE compresses images → latent features (8× smaller)
- Diffusion model operates on latents
- Decoder reconstructs high-quality images
- Faster training and inference

See [01_ARCHITECTURE.md](01_ARCHITECTURE.md#stage-2-latent-diffusion-model) for details.

### ControlNet

Add conditional control without retraining diffusion model:
- Copy of diffusion U-Net (frozen original + trainable copy)
- Zero convolutions for gradual integration
- Task-specific conditioning encoder
- Minimal retraining on labeled data

See [01_ARCHITECTURE.md](01_ARCHITECTURE.md#stage-3-controlnet) for details.

## Technical Stack

### Frameworks
- **PyTorch**: Deep learning framework
- **MONAI**: Medical imaging library
- **Weights & Biases**: Experiment tracking

### Key Models
- **AutoencoderKL**: Variational autoencoder
- **PatchDiscriminator**: GAN discriminator
- **U-Net**: Diffusion backbone
- **ControlNet**: Conditional control

### Schedulers
- **DDPM**: Denoising Diffusion Probabilistic Models (1000 steps)
- **DDIM**: Denoising Diffusion Implicit Models (50-100 steps, faster)

## Research Context

This project is based on the MAISI paper:

**Paper**: "MAISI: Medical AI for Synthetic Imaging"
- **Authors**: Pengfei Guo, Can Zhao, Dong Yang, et al.
- **Affiliations**: NVIDIA, NIH, University of Arkansas
- **arXiv**: 2409.11169v3
- **Published**: December 2025

### Key Contributions

1. **First framework** to generate realistic 3D CT images > 512³ voxels
2. **Tensor Splitting Parallelism** for memory-efficient generation
3. **Versatile conditioning** via ControlNet
4. **Strong empirical results**: FID 3.301 vs 98.208 for baselines
5. **Data augmentation**: 4-7% improvement in tumor segmentation

See [02_PAPER_SUMMARY.md](02_PAPER_SUMMARY.md) for comprehensive paper summary.

## Datasets

### Original MAISI Datasets

**VAE Training** (39,206 CT + 18,827 MRI):
- Multiple anatomical regions (chest, abdomen, head, neck)
- Diverse imaging protocols and patient demographics

**Diffusion Training** (10,277 CT volumes):
- 24 public datasets
- Various body regions and disease conditions

**ControlNet Training** (6,330 CT volumes):
- 127 anatomical structure annotations (TotalSegmentator)
- 5 tumor types (liver, pancreas, lung, colon, bone)

### This Repository's Datasets

**OCT Retinal Imaging**:
- KermanyV3 dataset (resized OCT scans)
- RETOUCH pathology annotations
- Multiple scanner modalities

## Performance Metrics

Based on MAISI paper results:

| Metric | MAISI | HA-GAN (Baseline) | Improvement |
|--------|-------|-------------------|-------------|
| FID (autoPET) | **3.301** | 98.208 | **96.5%** |
| Data Aug (DSC) | **+6.5%** | +2.2% (DiffTumor) | **+4.3%** |

See [02_PAPER_SUMMARY.md](02_PAPER_SUMMARY.md#experiments-and-results) for details.

## Common Pitfalls and Solutions

### Training Issues

1. **VAE produces blurry reconstructions**
   - Increase `perceptual_weight` (0.1 → 0.3)
   - Train longer (100 → 200 epochs)
   - See [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md#troubleshooting)

2. **Diffusion model poor quality**
   - Train longer (>500 epochs)
   - Use classifier-free guidance
   - Check VAE quality first

3. **ControlNet has no effect**
   - Verify zero convolutions initialized
   - Increase training epochs
   - Check conditioning preprocessing

### Inference Issues

1. **Samples don't match conditions**
   - Increase guidance scale (7.5 → 10.0)
   - Increase ControlNet scale (1.0 → 1.5)
   - See [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md#troubleshooting)

2. **Out of memory errors**
   - Reduce batch size
   - Use DDIM with fewer steps
   - Enable gradient checkpointing
   - Use tensor splitting

3. **Slow generation**
   - Switch DDPM (1000 steps) → DDIM (50 steps)
   - Use half precision
   - Use multiple GPUs for batches

## Additional Resources

### External Links

- **MAISI Paper**: https://arxiv.org/abs/2409.11169
- **NVIDIA MedTech**: https://github.com/NVIDIA/MedTech
- **MONAI Framework**: https://monai.io/
- **Weights & Biases**: https://wandb.ai/

### Related Papers

- Latent Diffusion Models: Rombach et al., CVPR 2022
- ControlNet: Zhang et al., ICCV 2023
- DDPM: Ho et al., NeurIPS 2020
- DDIM: Song et al., ICLR 2021

## Contributing

When modifying this codebase:

1. **Update documentation** if you change architecture or workflows
2. **Add configs** for new experiments to `configs/`
3. **Log experiments** to Weights & Biases
4. **Save checkpoints** with descriptive names
5. **Document new features** in appropriate `.md` files

## Maintenance

### Updating Documentation

If you make significant changes to the codebase:

1. Update relevant `.md` file in `ai-doc/`
2. Keep documentation in sync with code
3. Add examples for new features
4. Update troubleshooting sections

### Version Control

- Git tracks all changes
- Use descriptive commit messages
- Branch for major experiments
- Tag releases with version numbers

## License

See `LICENSE.weights` in root directory for model weights licensing.

Code is based on MONAI (Apache 2.0) and NVIDIA MedTech.

---

## Document Versions

- **Created**: February 7, 2026
- **Last Updated**: February 7, 2026
- **Version**: 1.0
- **Maintainer**: AI Documentation System

---

## Feedback

If you find errors or have suggestions for improving this documentation:
- Open an issue in the repository
- Submit a pull request with corrections
- Contact the project maintainers

---

**Happy Generating! 🚀**
