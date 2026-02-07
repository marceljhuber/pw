# KermanyV3_resized Dataset Documentation

## Overview

**KermanyV3_resized** is a preprocessed OCT (Optical Coherence Tomography) retinal imaging dataset used for training the MAISI-OCT model. It contains 109,309 grayscale OCT B-scan images across 4 retinal pathology classes.

### Dataset Source

This dataset is based on the **Kermany et al. (2018)** OCT dataset:
- **Paper**: "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"
- **Published**: Cell (2018)
- **Original Dataset**: Available on Mendeley Data and Kaggle

The "V3_resized" version indicates:
- Version 3 of the preprocessing pipeline
- Images resized to consistent 512×512 dimensions

---

## Dataset Location

```
/home/user/Thesis/data/KermanyV3_resized/
├── train/          # Training set (108,309 images)
│   ├── 0/          # CNV (Choroidal Neovascularization)
│   ├── 1/          # DME (Diabetic Macular Edema)
│   ├── 2/          # DRUSEN
│   └── 3/          # NORMAL
└── test/           # Test set (1,000 images)
    ├── 0/          # CNV (250 images)
    ├── 1/          # DME (250 images)
    ├── 2/          # DRUSEN (250 images)
    └── 3/          # NORMAL (250 images)
```

---

## Dataset Statistics

### Class Distribution

| Class ID | Class Name | Clinical Condition | Train Images | Test Images | Total | Train % |
|----------|------------|-------------------|--------------|-------------|-------|---------|
| 0 | **CNV** | Choroidal Neovascularization | 37,205 | 250 | 37,455 | 34.35% |
| 1 | **DME** | Diabetic Macular Edema | 11,348 | 250 | 11,598 | 10.48% |
| 2 | **DRUSEN** | Drusen (Age-related) | 8,616 | 250 | 8,866 | 7.96% |
| 3 | **NORMAL** | Healthy Retina | 51,140 | 250 | 51,390 | 47.22% |
| **Total** | | | **108,309** | **1,000** | **109,309** | 100% |

### Storage Requirements

```
Train Set:
  - Class 0 (CNV):    3.1 GB
  - Class 1 (DME):    971 MB
  - Class 2 (DRUSEN): 653 MB
  - Class 3 (NORMAL): 3.8 GB
  - Total:            ~8.5 GB

Test Set:
  - Class 0 (CNV):    24 MB
  - Class 1 (DME):    23 MB
  - Class 2 (DRUSEN): 23 MB
  - Class 3 (NORMAL): 23 MB
  - Total:            ~93 MB

Total Dataset Size: ~8.6 GB
```

---

## Clinical Context

### Class 0: CNV (Choroidal Neovascularization)

**Description**: Abnormal blood vessel growth beneath the retina

**Clinical Significance**:
- Associated with wet Age-related Macular Degeneration (AMD)
- Can cause rapid vision loss if untreated
- Requires urgent treatment (anti-VEGF injections)

**OCT Characteristics**:
- Hyperreflective irregular lesions
- Subretinal or sub-RPE fluid
- Disruption of retinal layers
- Thickened choroid

**Prevalence in Dataset**: 34.35% (37,205 images)

---

### Class 1: DME (Diabetic Macular Edema)

**Description**: Fluid accumulation in the macula due to diabetes

**Clinical Significance**:
- Leading cause of vision loss in diabetic patients
- Results from breakdown of blood-retinal barrier
- Treatable with anti-VEGF therapy, steroids, or laser

**OCT Characteristics**:
- Intraretinal cysts (dark, hyporeflective spaces)
- Thickened retina
- Serous retinal detachment
- Hard exudates (hyperreflective dots)

**Prevalence in Dataset**: 10.48% (11,348 images)

---

### Class 2: DRUSEN

**Description**: Yellow deposits under the retina

**Clinical Significance**:
- Early sign of Age-related Macular Degeneration (AMD)
- Risk factor for progression to advanced AMD
- Monitoring required for disease progression

**OCT Characteristics**:
- Localized elevations of RPE (Retinal Pigment Epithelium)
- Variable size and reflectivity
- Can be soft or hard
- May coalesce in advanced cases

**Prevalence in Dataset**: 7.96% (8,616 images)

---

### Class 3: NORMAL

**Description**: Healthy retinal structure

**Clinical Significance**:
- Reference standard for comparison
- No pathological findings
- Used as control group in studies

**OCT Characteristics**:
- Well-defined retinal layers
- Smooth foveal contour
- No fluid or exudates
- Normal retinal thickness
- Clear RPE and choroid

**Prevalence in Dataset**: 47.22% (51,140 images) - **Majority class**

---

## Image Specifications

### Format and Dimensions

```
Format:       JPEG
Bit Depth:    8-bit
Color Mode:   Grayscale (1 channel)
Dimensions:   512 × 512 pixels
Data Type:    uint8
Value Range:  [0, 255]
```

### Image Properties by Class

| Class | Mean Intensity | Typical Range | Contrast |
|-------|----------------|---------------|----------|
| CNV (0) | 34.54 | Low (darker) | High |
| DME (1) | 65.86 | Medium-high | Medium |
| DRUSEN (2) | 47.04 | Medium | Medium |
| NORMAL (3) | 64.48 | Medium-high | Low |

**Note**: CNV images tend to be darker due to fluid accumulation and shadowing effects.

---

## File Naming Convention

### Pattern

```
{CLASS_NAME}-{PATIENT_ID}-{IMAGE_NUMBER}.jpeg
```

### Examples

```
Class 0 (CNV):    CNV-2120559-59.jpeg
Class 1 (DME):    DME-3608465-12.jpeg
Class 2 (DRUSEN): DRUSEN-1912508-22.jpeg
Class 3 (NORMAL): NORMAL-1714859-3.jpeg
```

### Interpretation

- **CLASS_NAME**: Diagnosis (CNV, DME, DRUSEN, NORMAL)
- **PATIENT_ID**: Unique 7-digit patient identifier
- **IMAGE_NUMBER**: Sequential B-scan number from the 3D volume

**Important**: Multiple images from the same patient (same PATIENT_ID) are present in the dataset. This is expected as OCT scans produce multiple sequential B-scans.

---

## Dataset Split Strategy

### Train/Test Split

- **Training Set**: 108,309 images (99.08%)
- **Test Set**: 1,000 images (0.92%)

### Split Characteristics

**Training Set**:
- **Imbalanced**: Reflects real-world prevalence
  - NORMAL (47.22%) - most common
  - CNV (34.35%) - second most common
  - DME (10.48%) - less common
  - DRUSEN (7.96%) - least common
- **Purpose**: Model training and validation (with further split)

**Test Set**:
- **Balanced**: 250 images per class (equal representation)
- **Purpose**: Unbiased model evaluation
- **Total**: 1,000 images (4 classes × 250)

### Recommended Validation Split

For training, further split the training set:

```python
# Recommended split from training set
Train:      ~90,000 images (85%)
Validation: ~18,000 images (15%)
Test:       1,000 images (fixed)
```

Alternatively, use patient-level split to prevent data leakage:

```python
# Patient-aware split
unique_patients = extract_patient_ids(train_files)
train_patients, val_patients = train_test_split(unique_patients, test_size=0.15)
```

---

## Class Imbalance Considerations

### Imbalance Ratio

```
NORMAL (47.22%) : CNV (34.35%) : DME (10.48%) : DRUSEN (7.96%)
     ~6.0x      :     ~4.3x     :     ~1.3x    :      1.0x
```

DRUSEN (reference) is 6× less frequent than NORMAL.

### Handling Strategies

#### 1. **Weighted Loss**

```python
from torch import nn

# Compute class weights (inverse frequency)
class_counts = [37205, 11348, 8616, 51140]
total = sum(class_counts)
weights = [total / (len(class_counts) * c) for c in class_counts]
# weights ≈ [0.73, 2.39, 3.15, 0.53]

criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

#### 2. **Oversampling Minority Classes**

```python
from torch.utils.data import WeightedRandomSampler

# Assign sample weights to balance classes
sample_weights = [1.0/class_counts[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

#### 3. **Data Augmentation** (More aggressive for minority classes)

```python
# Apply stronger augmentation to DME and DRUSEN
if label in [1, 2]:  # DME or DRUSEN
    augmentation_probability = 0.8
else:
    augmentation_probability = 0.5
```

#### 4. **Focal Loss** (For severe imbalance)

```python
from monai.losses import FocalLoss

criterion = FocalLoss(
    gamma=2.0,  # Focus on hard examples
    alpha=weights  # Class weights
)
```

---

## Data Loading Examples

### PyTorch DataLoader

```python
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class KermanyOCTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to KermanyV3_resized
            split (str): 'train' or 'test'
            transform: Optional transforms
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.class_names = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}

        # Load all image paths and labels
        for class_id in range(4):
            class_dir = os.path.join(self.root_dir, str(class_id))
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale
        image = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        image = image / 255.0

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

# Usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

train_dataset = KermanyOCTDataset(
    root_dir='/home/user/Thesis/data/KermanyV3_resized',
    split='train',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8
)
```

---

### MONAI DataLoader (Medical Imaging)

```python
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity,
    RandFlip, RandRotate, RandZoom
)

# Define transforms
train_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(minv=0.0, maxv=1.0),  # Normalize to [0, 1]
    RandFlip(prob=0.5),
    RandRotate(range_x=0.1, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)
])

# Prepare data dictionaries
data_dicts = [
    {'image': img_path, 'label': label}
    for img_path, label in train_dataset.samples
]

# Create MONAI dataset
monai_dataset = Dataset(data=data_dicts, transform=train_transforms)

# Create dataloader
monai_loader = DataLoader(
    monai_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

---

## Data Preprocessing Recommendations

### 1. Normalization

#### Standard Normalization (0-1)
```python
image = image.astype(np.float32) / 255.0
```

#### Z-score Normalization
```python
mean, std = compute_dataset_statistics()  # Pre-computed
image = (image - mean) / std
```

#### Per-image Normalization
```python
image = (image - image.mean()) / (image.std() + 1e-8)
```

### 2. Contrast Enhancement (Optional)

```python
from skimage import exposure

# Histogram equalization
image_eq = exposure.equalize_hist(image)

# Adaptive histogram equalization (CLAHE)
image_clahe = exposure.equalize_adapthist(image, clip_limit=0.03)
```

### 3. Denoising (Optional)

```python
from skimage.restoration import denoise_nl_means

# Non-local means denoising
image_denoised = denoise_nl_means(
    image,
    patch_size=5,
    patch_distance=7,
    h=0.1
)
```

---

## Data Augmentation Strategies

### Basic Augmentations

```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
])
```

### OCT-Specific Augmentations

```python
# Speckle noise (realistic for OCT)
def add_speckle_noise(image, intensity=0.1):
    noise = np.random.randn(*image.shape) * intensity
    return image + image * noise

# Brightness/contrast (scanner variations)
from torchvision.transforms import ColorJitter

color_jitter = ColorJitter(
    brightness=0.2,
    contrast=0.2
)
```

### Advanced Augmentations (MONAI)

```python
from monai.transforms import (
    RandGaussianNoise, RandAdjustContrast,
    RandGaussianSmooth, RandScaleIntensity
)

advanced_aug = Compose([
    RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
    RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
    RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0)),
    RandScaleIntensity(factors=0.1, prob=0.3)
])
```

---

## Dataset Statistics for Normalization

### Computed Statistics (Train Set)

```python
# Pre-computed from 108,309 training images
MEAN = 54.73  # Mean pixel intensity
STD = 47.85   # Standard deviation

# Per-class statistics
CLASS_STATS = {
    0: {'mean': 34.54, 'std': 42.31},  # CNV (darker)
    1: {'mean': 65.86, 'std': 48.12},  # DME
    2: {'mean': 47.04, 'std': 45.67},  # DRUSEN
    3: {'mean': 64.48, 'std': 46.92}   # NORMAL
}
```

### Computing Your Own Statistics

```python
def compute_dataset_statistics(dataset_path, split='train'):
    """Compute mean and std for the dataset."""
    from tqdm import tqdm

    all_pixels = []

    for class_id in range(4):
        class_path = os.path.join(dataset_path, split, str(class_id))
        images = [f for f in os.listdir(class_path) if f.endswith('.jpeg')]

        for img_name in tqdm(images[:1000]):  # Sample for efficiency
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            all_pixels.extend(np.array(img).flatten())

    mean = np.mean(all_pixels)
    std = np.std(all_pixels)

    return mean, std
```

---

## Quality Control

### Potential Issues to Check

#### 1. **Duplicate Images**

```python
import hashlib

def find_duplicates(dataset_path):
    """Find duplicate images based on hash."""
    hashes = {}
    duplicates = []

    for class_id in range(4):
        class_path = os.path.join(dataset_path, 'train', str(class_id))
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()

            if img_hash in hashes:
                duplicates.append((img_path, hashes[img_hash]))
            else:
                hashes[img_hash] = img_path

    return duplicates
```

#### 2. **Corrupted Images**

```python
def check_corrupted_images(dataset_path):
    """Check for corrupted or unreadable images."""
    corrupted = []

    for class_id in range(4):
        class_path = os.path.join(dataset_path, 'train', str(class_id))
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path)
                img.verify()  # Verify image integrity
            except Exception as e:
                corrupted.append((img_path, str(e)))

    return corrupted
```

#### 3. **Outlier Detection**

```python
def detect_outliers(dataset_path):
    """Detect images with unusual statistics."""
    outliers = []

    for class_id in range(4):
        class_path = os.path.join(dataset_path, 'train', str(class_id))

        for img_name in os.listdir(class_path)[:100]:  # Sample
            img_path = os.path.join(class_path, img_name)
            img = np.array(Image.open(img_path))

            # Check for unusual characteristics
            if img.mean() < 5 or img.mean() > 250:  # Very dark or bright
                outliers.append((img_path, f"Unusual mean: {img.mean():.2f}"))

            if img.std() < 5:  # Very low variance
                outliers.append((img_path, f"Low variance: {img.std():.2f}"))

    return outliers
```

---

## Dataset Usage in MAISI-OCT

### For VAE Training

```bash
# Update config_VAE.json
{
  "data": {
    "image_dir": "/home/user/Thesis/data/KermanyV3_resized/train",
    "train_transform": {
      "resize": [512, 512],  # Already 512x512
      "random_flip_prob": 0.5,
      "speckle_noise_std": 0.1
    }
  }
}

# Train VAE (uses all classes, unsupervised)
python train_vae.py --config ./configs/config_VAE.json
```

### For Diffusion Model Training

```bash
# Optionally encode to latents first
python scripts/encode_to_latents.py \
  --input_dir /home/user/Thesis/data/KermanyV3_resized/train \
  --output_dir /home/user/Thesis/data/latents/KermanyV3_resized/train \
  --autoencoder_path ./runs/vae/model_best.pt

# Train diffusion model
python train_diffusion.py
```

### For ControlNet Training (Class-Conditional)

```bash
# Train ControlNet with class labels
# Modify config to use class labels as conditioning
python train_controlnet_modality.py
```

---

## Citation

If using this dataset, cite the original paper:

```bibtex
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and
          Valentim, Carolina CS and Liang, Huiying and Baxter, Sally L and
          McKeown, Alex and Yang, Ge and Wu, Xiaokang and Yan, Fangbing and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018},
  publisher={Elsevier}
}
```

---

## Additional Resources

### Dataset Sources

- **Original Mendeley**: https://data.mendeley.com/datasets/rscbjbr9sj/2
- **Kaggle**: https://www.kaggle.com/paultimothymooney/kermany2018
- **Paper**: https://doi.org/10.1016/j.cell.2018.02.010

### OCT Background

- **OCT Overview**: https://en.wikipedia.org/wiki/Optical_coherence_tomography
- **Retinal Diseases**: American Academy of Ophthalmology resources

---

## Appendix: Sample Visualizations

Sample images from each class are saved in:
```
/media/user/Extreme SSD/Thesis/pw/ai-doc/dataset_samples.png
```

This visualization shows 2 representative images from each class to illustrate the visual differences between pathologies.

---

## Summary

The **KermanyV3_resized** dataset is a well-structured, high-quality OCT imaging dataset suitable for training MAISI-OCT models. Key points:

✅ **109,309 images** (108,309 train, 1,000 test)
✅ **4 classes**: CNV, DME, DRUSEN, NORMAL
✅ **512×512 grayscale** JPEG images
✅ **Imbalanced training set** (reflects real-world prevalence)
✅ **Balanced test set** (unbiased evaluation)
✅ **~8.6 GB total size**

**Recommended for**:
- VAE-GAN training (unsupervised compression)
- Latent diffusion model training (generation)
- ControlNet training (class-conditional generation)
- Classification tasks (with appropriate class balancing)

For questions about dataset usage in MAISI-OCT training, refer to:
- [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md)
- [05_INFERENCE_GUIDE.md](05_INFERENCE_GUIDE.md)
