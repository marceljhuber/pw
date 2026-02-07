#!/bin/bash
# Quick setup script for MAISI fast training environment

set -e  # Exit on error

echo "===== Setting up MAISI Fast Training Environment ====="

# Activate conda
source /home/user/miniconda3/etc/profile.d/conda.sh

# Create environment if it doesn't exist
if ! conda env list | grep -q "maisi"; then
    echo "Creating conda environment..."
    conda create -n maisi python=3.10 -y
fi

# Activate environment
conda activate maisi

# Install PyTorch with CUDA 11.8 (compatible with driver 535)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy scipy scikit-learn scikit-image pandas matplotlib seaborn tqdm pillow opencv-python

# Install medical imaging libraries
echo "Installing medical imaging libraries..."
pip install monai==1.3.0 nibabel

# Install diffusion models
echo "Installing diffusion libraries..."
pip install diffusers==0.25.0 accelerate transformers

# Install training utilities
echo "Installing training utilities..."
pip install wandb tensorboard

# Install additional utilities
echo "Installing additional utilities..."
pip install albumentations lpips einops fire pyyaml

# Test CUDA availability
echo "===== Testing PyTorch and CUDA ====="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "===== Environment setup complete! ====="
echo "To activate: conda activate maisi"
