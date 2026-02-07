# MAISI Paper Summary

**Title**: MAISI: Medical AI for Synthetic Imaging
**Authors**: Pengfei Guo, Can Zhao, Dong Yang, et al.
**Affiliations**: NVIDIA, National Institutes of Health, University of Arkansas for Medical Sciences
**arXiv**: 2409.11169v3 [eess.IV]
**Date**: December 2025

## Abstract

Medical imaging analysis faces three critical challenges:
1. **Data scarcity** - Limited datasets for rare diseases
2. **High annotation costs** - Expert annotations are expensive and time-consuming
3. **Privacy concerns** - Patient data sharing restrictions

**MAISI** proposes a novel framework for generating high-resolution 3D CT volumes (up to **512 × 512 × 768 voxels**) to address these challenges through synthetic data generation.

## Key Contributions

### 1. Novel Framework for 3D Medical Image Synthesis
- **First** framework to generate realistic 3D CT images larger than 512³ voxels
- Combines foundation models (VAE, Latent Diffusion) with ControlNet for versatile generation
- Trained on **10,277 CT volumes** from **24 diverse datasets**

### 2. Tensor Splitting Parallelism (TSP)
- Innovative technique to overcome GPU memory limitations
- Enables generation of ultra-high-resolution volumes
- Distributes computation across multiple devices efficiently

### 3. Versatile Conditional Generation
- **127 anatomical structures** for fine-grained control
- Flexible volume dimensions and voxel spacing
- Task-specific adaptations without extensive retraining

### 4. Strong Empirical Results
- **FID**: 3.301 (vs 98.208 for HA-GAN baseline)
- **Data Augmentation**: 4-7% improvement in tumor segmentation DSC
- **Diverse Applications**: CT generation, tumor inpainting, segmentation masks

## Methodology

### Three-Stage Architecture

#### **Stage 1: VAE-GAN Volume Compression**

Trains a Variational Autoencoder with adversarial loss on **39,206 CT + 18,827 MRI volumes**.

**Objective Function**:
```
ℒ_AE = min/max (ℒ_recon + ℒ_lpips + ℒ_adv + ℒ_KL + ℒ_adv)
        E,D,Disc
```

Components:
- **ℒ_recon**: L1 reconstruction loss
- **ℒ_lpips**: Perceptual loss (SqueezeNet features)
- **ℒ_adv**: Adversarial loss (PatchGAN discriminator)
- **ℒ_KL**: KL divergence regularization
- **ℒ_adv**: Adversarial weight term

**Architecture Details**:
- Input: Grayscale CT slices (H × W × D)
- Encoder: 3 downsampling stages → 4-channel latent space
- Decoder: 3 upsampling stages → reconstructed image
- Discriminator: 3-layer PatchGAN

**Training**:
- Batch size: 8 (V100 32GB GPU)
- Learning rate: 1e-4
- Epochs: 100
- Data augmentation: Random cropping, flipping, rotation, intensity scaling

**Key Innovation - Tensor Splitting**:
- Feature maps split into overlapping segments
- Each segment processed independently or across GPUs
- Results stitched using normalization layers
- Reduces peak memory while maintaining quality

#### **Stage 2: Latent Diffusion Model**

Trained on **10,277 CT volumes** from diverse public datasets.

**Diffusion Process**:

Forward (add noise):
```
q(z_t | z_0) = 𝒩(z_t; √ᾱ_t z_0, (1-ᾱ_t)I)
```

Reverse (denoise):
```
Training: ℒ = 𝔼[‖ε - ε_θ(z_t, t, c_p)‖₁]
```

**Conditioning Mechanisms**:
- **Body Region** (c_top, c_bottom): One-hot vectors for head-neck, chest, abdomen, pelvis
- **Voxel Spacing** (s): [s_x, s_y, s_z] in mm
- U-Net backbone conditioned on timestep t and conditions c_p

**U-Net Architecture**:
- Channels: [64, 128, 256, 512]
- Attention levels: [False, False, True, True]
- Number of residual blocks: 2 per level
- Flash attention for efficiency

**Noise Scheduler**:
- Type: DDPM (Denoising Diffusion Probabilistic Models)
- Training timesteps: 1,000
- Beta schedule: Scaled linear (0.0015 → 0.0195)
- Inference: Can use DDIM for faster sampling

**Datasets** (24 total):
- AbdomenCT-1K, AeroPath, AMOS22, autoPET23
- Bone-Lesion, BTCV, COVID-19, CRLM-CT
- CT-ORG, CTPelvic1K-CLINIC, LIDC
- MSD (Task03, 06, 07, 08, 09, 10)
- Multi-organ-Abdominal-CT, NLST, Pancreas-CT
- StonyBrook-CT, TCIA Colon, TotalSegmentatorV2, VerSe

#### **Stage 3: ControlNet**

Enables task-specific control without retraining the diffusion model.

**Architecture**:
1. **Trainable ControlNet copy** of the diffusion U-Net
2. **Frozen diffusion model** (preserves learned knowledge)
3. **Zero convolutions** connecting the two (start at zero influence)

**Training Objective**:
```
ℒ = 𝔼[‖ε - ε_θ(z_t, t, c_p, c_T)‖₁]
```
where c_T is task-specific conditioning from ControlNet.

**Supported Tasks**:

1. **MAISI CT Generation**
   - Input: 127 anatomical structure masks (TotalSegmentator)
   - Dataset: 6,330 CT volumes with segmentation annotations
   - Use case: Generate CT with specific anatomy

2. **MAISI Tumor Inpainting**
   - Input: Tumor masks (5 types: liver, pancreas, lung, colon, bone)
   - Method: Augment real patient data with synthetic tumors
   - Use case: Data augmentation for rare tumor types

**Training Strategy**:
- Phase 1: Train diffusion model on unlabeled data (10,277 volumes)
- Phase 2: Train ControlNet on labeled subset (6,330 volumes)
- Only ControlNet parameters updated; diffusion model frozen

### Tensor Splitting Parallelism (TSP) Details

**Problem**: Standard 3D convolutions exceed GPU memory for large volumes

**Solution**: TSP partitions feature maps along spatial dimensions

**Process**:
1. Split input into K segments with overlaps
2. Assign each segment to a device (or process sequentially)
3. Apply convolution to each segment independently
4. Stitch outputs using group normalization layers

**Mathematical Formulation**:
```
For feature map F of shape (C, H, W, D):
1. Partition along dimension d into K segments: F_1, ..., F_K
2. Process: O_k = Conv(F_k) for k = 1, ..., K
3. Concatenate: O = Concat(O_1, ..., O_K)
4. Normalize: Output = GroupNorm(O)
```

**Benefits**:
- Reduces peak memory from O(C×H×W×D) to O(C×H×W×D/K)
- Enables 512³+ volume generation on consumer GPUs
- Minimal quality degradation

## Experiments and Results

### 4.1 Datasets

**Volume Compression (VAE)**:
- Training: 37,243 CT + 17,887 MRI volumes
- Validation: 1,963 CT + 940 MRI volumes
- Regions: Chest, abdomen, head, neck

**Latent Diffusion Model**:
- Training: 10,277 CT volumes (24 datasets)
- Covers: Various body regions, diseases, imaging protocols

**ControlNet**:
- MAISI CT Generation: 6,330 volumes with 127-structure annotations
- MAISI Inpainting: 5 tumor types with segmentation masks

### 4.2 Evaluation: VAE-GAN

**Metrics**:
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

**Results** (Table 1 in paper):
| Dataset    | Model          | LPIPS ↓ | SSIM ↑ | PSNR ↑ | GPU Hours |
|------------|----------------|---------|--------|--------|-----------|
| MSD Task07 | MAISI VAE      | 0.038   | 0.978  | 37.266 | 0h        |
|            | Dedicated VAE  | 0.047   | 0.971  | 34.750 | 307h      |
| MSD Task08 | MAISI VAE      | 0.046   | 0.970  | 36.539 | 0h        |
|            | Dedicated VAE  | 0.041   | 0.974  | 37.118 | 367h      |
| BraTS18    | MAISI VAE      | 0.026   | 0.977  | 39.003 | 0h        |
|            | Dedicated VAE  | 0.048   | 0.975  | 38.911 | 62.3h     |

**Key Finding**: MAISI VAE achieves comparable quality to dedicated models without additional training, demonstrating foundational model effectiveness.

### 4.3 Evaluation: Diffusion Model

#### Synthesis Quality

**Fréchet Inception Distance (FID)** on autoPET 2023:

| Method      | FID ↓  |
|-------------|--------|
| **MAISI DM**| **3.301** |
| LDM         | 10.191 |
| HA-GAN      | 98.208 |

**FID Across Anatomical Views**:
| View     | Axial  | Sagittal | Coronal | Average |
|----------|--------|----------|---------|---------|
| DDPM     | 18.524 | 23.996   | 25.605  | 22.608  |
| LDM      | 16.353 | 10.191   | 10.093  | 12.379  |
| HA-GAN   | 17.832 | 10.586   | 13.372  | 13.777  |
| **MAISI**| **3.301** | **5.838** | **9.109** | **6.083** |

**Qualitative Comparison** (Figure 4):
- MAISI generates more detailed anatomical structures
- Better global coherence across slices
- Fewer artifacts compared to baselines

#### Response to Primary Conditions

**Body Region Control** (Figure 5):
- Successfully generates anatomically consistent images for:
  - Different body regions (chest, abdomen, pelvis)
  - Variable voxel spacing (0.5×, 1×, 1.5× original)
  - Flexible output dimensions (256×256, 384×384, 512×512)

**Observations**:
- Smooth scaling without boundary artifacts
- Anatomically plausible expansion/contraction
- Maintains image quality across conditions

### 4.4 Data Augmentation Results

**Setup**: Train tumor segmentation models with/without MAISI-generated data

**Datasets**:
- MSD Task03 (Liver Tumor) - 103 real samples
- MSD Task06 (Lung Tumor) - baseline
- MSD Task07 (Pancreas Tumor) - baseline
- MSD Task08 (Liver Tumor) - baseline
- MSD Task10 (Colon Tumor) - baseline

**Augmentation Strategy**:
- Generate synthetic images with tumors using ControlNet
- Mix with real data at ratios: 1:0, 1:0.5, 1:1, 1:1.5

**Metrics**: Dice Similarity Coefficient (DSC)

**Results** (Figure 6):

| Tumor Type | Real Only | DiffTumor | MAISI Generation | MAISI Inpainting | Improvement |
|------------|-----------|-----------|------------------|------------------|-------------|
| Bone       | 0.504     | -         | 0.539            | -                | **+3.6%**   |
| Liver (03) | 0.662     | 0.688     | 0.714            | -                | **+5.1%**   |
| Lung (05)  | 0.581     | -         | 0.635            | 0.649            | **+6.9%**   |
| Pancreas   | 0.511     | -         | 0.507            | -                | **+7.9%**   |
| Colon      | 0.449     | -         | 0.485            | -                | **+3.6%**   |

**Key Findings**:
1. **Consistent improvements** across all tumor types
2. **MAISI Inpainting** (6.5% avg improvement) outperforms **MAISI Generation** (4.9%)
3. Larger improvements for difficult/small tumors (pancreas, lung)
4. All improvements **statistically significant** (Wilcoxon signed-rank test)

**Comparison to DiffTumor**:
- MAISI Inpainting: **6.5%** improvement for liver tumor
- DiffTumor: **2.2%** improvement for liver tumor
- MAISI demonstrates superior augmentation capability

### 4.5 ControlNet Applications

#### MAISI CT Generation

**Input**: 127 anatomical structure segmentation masks
**Output**: Realistic CT volumes matching the anatomy

**Examples** (Figure S3 - Supplementary):
- Bone lesion generation with accurate skeletal structure
- Liver tumor in correct anatomical position
- Lung nodule with realistic lung parenchyma
- Pancreas tumor preserving organ boundaries
- Colon tumor with proper intestinal context

**Evaluation**:
- Visual inspection shows anatomical consistency
- Structures maintain spatial relationships
- Disease features (tumors) integrate naturally

#### MAISI Tumor Inpainting

**Input**: Real CT + synthetic tumor mask
**Output**: CT with realistically integrated tumor

**Process**:
1. Select healthy patient CT
2. Generate tumor mask using DiffTumor/similar method
3. MAISI Inpainting blends tumor into image
4. Result: Augmented training data

**Advantages over CT Generation**:
- Preserves real patient diversity (age, sex, ethnicity)
- Maintains scanner-specific characteristics
- Only synthesizes pathology, not entire anatomy
- Better performance in data augmentation (+6.5% vs +4.9%)

### 4.6 Segmentation Evaluation on Synthetic Data

**Experiment**: Train segmentation models on synthetic data, test on real data

**Results** (Table S5 - Supplementary):

| Organ         | Real Data DSC | Synthetic Data DSC | Performance Gap |
|---------------|---------------|--------------------|-----------------|
| Liver         | 0.93          | 0.93               | 0.0%            |
| Spleen        | 0.94          | 0.93               | -1.1%           |
| Left Kidney   | 0.95          | 0.95               | 0.0%            |
| Right Kidney  | 0.95          | 0.95               | 0.0%            |
| Stomach       | 0.90          | 0.88               | -2.2%           |
| Gallbladder   | 0.75          | 0.77               | +2.7%           |
| Esophagus     | 0.76          | 0.71               | -6.6%           |
| Pancreas      | 0.80          | 0.70               | -12.5%          |
| Duodenum      | 0.69          | 0.54               | -21.7%          |
| Small Bowel   | 0.80          | 0.74               | -7.5%           |
| Bladder       | 0.87          | 0.86               | -1.1%           |

**Observations**:
- **Large organs** (liver, spleen, kidneys): Comparable performance
- **Small organs** (pancreas, duodenum): Performance gap remains
- Indicates need for improvement in small structure generation

## Discussion and Limitations

### Strengths

1. **Scalability**: Handles ultra-high-resolution 3D volumes
2. **Versatility**: Flexible conditioning for diverse tasks
3. **Foundation Model**: Generalizes across datasets without retraining
4. **Data Efficiency**: Improves downstream tasks with augmentation

### Limitations

1. **Computational Cost**: Generation is resource-intensive
   - Diffusion inference requires many steps (1000 for DDPM, 50+ for DDIM)
   - Not accessible to researchers with limited GPUs

2. **Small Organ Quality**: Performance gap for small structures
   - Pancreas, duodenum, small bowel less accurately generated
   - May require higher resolution or specialized architectures

3. **Demographic Representation**: Need to verify synthetic data captures diversity
   - Age, ethnicity, gender variations
   - Disease prevalence across populations

4. **Ethical Considerations**: Synthetic data societal impacts
   - Potential misuse for generating misleading medical images
   - Need for watermarking and detection mechanisms

### Future Directions

1. **Improve Accessibility**: Optimize for lower-resource settings
2. **Enhanced Small Structure Generation**: Specialized models or loss functions
3. **Diversity Validation**: Ensure synthetic data represents real-world populations
4. **Ethical Safeguards**: Develop detection and attribution methods

## Conclusion

MAISI represents a significant advancement in medical image synthesis:

- **First** to generate realistic 3D CT volumes larger than 512³ voxels
- **Versatile** framework adaptable to multiple tasks
- **Strong empirical results** on synthesis quality and downstream task improvement
- **Foundation model** approach reduces need for task-specific retraining

The framework demonstrates promising potential for addressing data scarcity, privacy concerns, and annotation costs in medical imaging, while highlighting important directions for future research.

## Citation

```bibtex
@article{guo2024maisi,
  title={MAISI: Medical AI for Synthetic Imaging},
  author={Guo, Pengfei and Zhao, Can and Yang, Dong and Xu, Ziyue and
          Nath, Vishwesh and Tang, Yucheng and Simon, Benjamin and
          Belue, Mason and Harmon, Stephanie and Turkbey, Baris and Xu, Daguang},
  journal={arXiv preprint arXiv:2409.11169},
  year={2024}
}
```

## Code and Resources

- **Paper**: https://arxiv.org/abs/2409.11169
- **NVIDIA MedTech**: https://github.com/NVIDIA/MedTech (official code)
- **NVIDIA NIM**: https://www.nvidia.com/en-us/nim/ (online demo)
- **MONAI**: https://monai.io/ (framework used)
