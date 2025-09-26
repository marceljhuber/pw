#!/usr/bin/env python3
"""
Conditional MAISI Diffusion Model Inference Script
Generates medical images conditioned on class labels: CNV, DME, DRUSEN, NORMAL

Usage:
    python conditional_inference.py --config config.json --checkpoint model.pt --output_dir results/ --samples 10,5,8,12
    python conditional_inference.py --config config.json --checkpoint model.pt --class_name CNV --num_samples 5
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from monai.transforms import SaveImage
from scripts.utils import define_instance
import logging

# Import your custom modules
from networks.conditional_maisi_wrapper import ConditionalMAISIWrapper
from networks.autoencoderkl_maisi import AutoencoderKlMaisi

# Class mapping
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_MAPPING = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Conditional MAISI Diffusion Model Inference")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file used during training"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory to save generated images"
    )

    # Option 1: Generate specific numbers for each class
    parser.add_argument(
        "--samples",
        type=str,
        help="Number of samples per class as comma-separated values (e.g., '10,5,8,12' for CNV,DME,DRUSEN,NORMAL)"
    )

    # Option 2: Generate for specific class
    parser.add_argument(
        "--class_name",
        type=str,
        choices=CLASS_NAMES,
        help="Generate images for specific class only"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate (used with --class_name)"
    )

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of denoising steps"
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (higher = more class-specific)"
    )

    parser.add_argument(
        "--use_classifier_free_guidance",
        action="store_true",
        help="Enable classifier-free guidance"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (increase if you have enough GPU memory)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )

    parser.add_argument(
        "--use_fp32",
        action="store_true",
        help="Force float32 precision (fixes mixed precision issues)"
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Merge configs with priority handling
    merged_config = {}
    for section in ["main", "model_config", "env_config", "vae_def", "conditional_config", "model"]:
        if section in config:
            merged_config.update(config[section])

    return argparse.Namespace(**merged_config)


def safe_load_checkpoint(checkpoint_path, device, logger):
    """Safely load checkpoint with MONAI compatibility"""
    try:
        # First try with weights_only=False for MONAI compatibility
        logger.info("Loading checkpoint (MONAI compatible mode)...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info("✅ Checkpoint loaded successfully")
        return checkpoint
    except Exception as e:
        logger.warning(f"⚠️  Failed to load checkpoint: {e}")
        logger.warning("Attempting alternative loading method...")

        try:
            # Alternative: try to add safe globals for MONAI
            from monai.utils.enums import MetaKeys
            torch.serialization.add_safe_globals([MetaKeys])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            logger.info("✅ Checkpoint loaded with safe globals")
            return checkpoint
        except Exception as e2:
            logger.error(f"❌ Could not load checkpoint: {e2}")
            logger.error("Using empty checkpoint - model will have random weights")
            return {}


def fix_mixed_precision_model(model, use_fp32=False):
    """Fix mixed precision issues in MONAI models"""
    if use_fp32:
        # Convert entire model to float32
        model = model.float()
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.float()
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.float()
    else:
        # Ensure consistent precision across all parameters
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_dtype = module.weight.dtype
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(weight_dtype)

    return model


def load_models(config, checkpoint_path, device, logger, use_fp32=False):
    """Load the trained diffusion model and autoencoder"""
    logger.info("Loading models...")

    # Get conditional config parameters
    conditional_config = getattr(config, 'conditional_config', {})
    if hasattr(config, 'conditional_config'):
        conditional_config = config.conditional_config
    else:
        # Fallback to individual attributes if they exist
        conditional_config = {
            'num_classes': getattr(config, 'num_classes', 4),
            'class_emb_dim': getattr(config, 'class_emb_dim', 64),
            'conditioning_method': getattr(config, 'conditioning_method', 'input_concat')
        }

    # Load conditional diffusion UNet
    logger.info("Loading conditional MAISI UNet...")
    unet = ConditionalMAISIWrapper(
        config_args=config,
        num_classes=conditional_config.get('num_classes', 4),
        class_emb_dim=conditional_config.get('class_emb_dim', 64),
        conditioning_method=conditional_config.get('conditioning_method', 'input_concat')
    ).to(device)

    # Load checkpoint safely
    checkpoint = safe_load_checkpoint(checkpoint_path, device, logger)

    # Load UNet state dict
    if "unet_state_dict" in checkpoint:
        try:
            unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
            logger.info("✅ Conditional UNet loaded from checkpoint")
        except Exception as e:
            logger.warning(f"⚠️  Could not load UNet state dict: {e}")
            logger.warning("Using random UNet weights")
    else:
        logger.warning("⚠️  No 'unet_state_dict' found in checkpoint, using random weights")

    # Fix mixed precision issues in UNet
    unet = fix_mixed_precision_model(unet, use_fp32)
    unet.eval()

    # Load noise scheduler
    logger.info("Loading noise scheduler...")
    noise_scheduler = define_instance(config, "noise_scheduler")

    # Load autoencoder (VAE) using config directly
    logger.info("Loading autoencoder...")

    # Get autoencoder config from the loaded config
    if hasattr(config, 'model') and 'autoencoder' in config.model:
        autoencoder_config = config.model['autoencoder']
    elif hasattr(config, 'autoencoder'):
        autoencoder_config = config.autoencoder
    else:
        # Fallback: extract from vae_def section
        autoencoder_config = {
            "spatial_dims": getattr(config, 'spatial_dims', 2),
            "in_channels": getattr(config, 'image_channels', 1),
            "out_channels": getattr(config, 'image_channels', 1),
            "latent_channels": getattr(config, 'latent_channels', 4),
            "num_channels": [64, 128, 256],
            "num_res_blocks": [2, 2, 2],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": False,  # Disable float16 to avoid precision issues
            "num_splits": 1,
            "dim_split": 1
        }

    # Force disable float16 normalization to avoid precision issues
    if 'norm_float16' in autoencoder_config:
        autoencoder_config['norm_float16'] = False

    autoencoder = AutoencoderKlMaisi(**autoencoder_config).to(device)

    # Load autoencoder checkpoint
    if hasattr(config, 'trained_autoencoder_path') and config.trained_autoencoder_path:
        try:
            vae_checkpoint = safe_load_checkpoint(config.trained_autoencoder_path, device, logger)
            if "autoencoder_state_dict" in vae_checkpoint:
                autoencoder.load_state_dict(vae_checkpoint["autoencoder_state_dict"])
                logger.info("✅ Autoencoder loaded from checkpoint")
            else:
                logger.warning("⚠️  No 'autoencoder_state_dict' found in VAE checkpoint")
        except Exception as e:
            logger.warning(f"⚠️  Could not load autoencoder checkpoint: {e}")
            logger.warning("Using random autoencoder weights - results may be poor!")
    else:
        logger.warning("⚠️  No autoencoder checkpoint path provided - using random weights")

    # Fix mixed precision issues in autoencoder
    autoencoder = fix_mixed_precision_model(autoencoder, use_fp32)
    autoencoder.eval()

    if use_fp32:
        logger.info("🔧 Using float32 precision for inference")
    else:
        logger.info("🔧 Using mixed precision with consistency fixes")

    # Get scale factor
    scale_factor = checkpoint.get("scale_factor", 1.0) if "scale_factor" in checkpoint else 1.0
    logger.info(f"Using scale factor: {scale_factor}")

    return unet, autoencoder, noise_scheduler, scale_factor


def conditional_sample_batch(
    unet,
    autoencoder,
    noise_scheduler,
    scale_factor,
    device,
    class_labels,
    latent_shape,
    num_inference_steps=1000,
    guidance_scale=7.5,
    use_cfg=True,
    use_fp32=False
):
    """
    Generate a batch of images with class conditioning

    Args:
        unet: Trained conditional UNet
        autoencoder: Trained VAE
        noise_scheduler: Noise scheduler
        scale_factor: Latent scaling factor
        device: Device to run on
        class_labels: List/tensor of class labels
        latent_shape: Shape of latent space [batch_size, channels, H, W] for 2D
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        use_cfg: Whether to use classifier-free guidance
        use_fp32: Whether to use float32 precision
    """
    batch_size = len(class_labels) if isinstance(class_labels, (list, tuple)) else class_labels.shape[0]

    # Create random noise with appropriate dtype
    dtype = torch.float32 if use_fp32 else torch.float16
    noise = torch.randn(batch_size, *latent_shape[1:], device=device, dtype=dtype)

    # Convert class labels to tensor if needed
    if isinstance(class_labels, (list, tuple)):
        class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)

    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    # Denoising loop
    for t in tqdm(noise_scheduler.timesteps, desc="Denoising"):
        timestep_tensor = t.unsqueeze(0).expand(batch_size).to(device)

        with torch.no_grad():
            if use_cfg and guidance_scale > 1.0:
                # Classifier-free guidance
                # Conditional prediction
                noise_pred_cond = unet(noise, timestep_tensor, class_labels=class_labels)

                # Unconditional prediction
                noise_pred_uncond = unet(noise, timestep_tensor, class_labels=None)

                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard conditional prediction
                noise_pred = unet(noise, timestep_tensor, class_labels=class_labels)

            # Ensure consistent dtype
            if noise_pred.dtype != noise.dtype:
                noise_pred = noise_pred.to(noise.dtype)

        # Handle the scheduler step output properly
        step_output = noise_scheduler.step(noise_pred, t, noise)

        # Different schedulers return different formats
        if hasattr(step_output, 'prev_sample'):
            # New format (SchedulerOutput object)
            noise = step_output.prev_sample
        elif isinstance(step_output, tuple):
            # Old format (tuple)
            noise = step_output[0]  # prev_sample is usually the first element
        else:
            # Fallback - assume it's the tensor directly
            noise = step_output

        # Ensure consistent dtype
        if hasattr(noise, 'dtype') and noise.dtype != dtype:
            noise = noise.to(dtype)

    # Decode latents to images with proper dtype handling
    with torch.no_grad():
        # Ensure noise has correct dtype before passing to decoder
        decode_input = (noise / scale_factor).to(dtype)
        synthetic_images = autoencoder.decode(decode_input)

        # Convert to float32 for post-processing
        synthetic_images = synthetic_images.float()

    return synthetic_images


def generate_images_for_class(
    unet, autoencoder, noise_scheduler, scale_factor, device,
    class_idx, class_name, num_samples, output_dir, args, logger
):
    """Generate images for a specific class"""
    logger.info(f"🎯 Generating {num_samples} images for class {class_idx} ({class_name})")

    # Create class-specific output directory
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    # Define latent shape based on config (2D MAISI for your case)
    # Adjust these dimensions based on your input image size and VAE downsampling factor
    latent_shape = [1, 4, 32, 32]  # [batch_size, channels, H, W] for 2D MAISI

    # Generate images in batches
    batch_size = min(args.batch_size, num_samples)
    generated_images = []

    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)

        # Create class labels for current batch
        class_labels = [class_idx] * current_batch_size

        # Update latent shape for current batch
        current_latent_shape = [current_batch_size] + latent_shape[1:]

        logger.info(f"Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")

        try:
            # Generate images
            synthetic_images = conditional_sample_batch(
                unet=unet,
                autoencoder=autoencoder,
                noise_scheduler=noise_scheduler,
                scale_factor=scale_factor,
                device=device,
                class_labels=class_labels,
                latent_shape=current_latent_shape,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_cfg=args.use_classifier_free_guidance,
                use_fp32=args.use_fp32
            )

            # Process and save images
            for j in range(current_batch_size):
                img_idx = i + j

                # Denormalize from [-1,1] to [0,1]
                image = (synthetic_images[j] + 1) / 2.0
                image = torch.clamp(image, 0, 1)

                # Convert to uint8 range [0,255]
                image = (image * 255).type(torch.uint8)

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_saver = SaveImage(
                    output_dir=class_output_dir,
                    output_postfix=f"{class_name}_{img_idx:04d}_{timestamp}",
                    output_ext=".png",
                    separate_folder=False,
                    print_log=False
                )
                img_saver(image)

                generated_images.append(image)

        except Exception as e:
            logger.error(f"❌ Error generating batch {i//batch_size + 1}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Continuing with next batch...")
            continue

    logger.info(f"✅ Generated {len(generated_images)} images for {class_name} in {class_output_dir}")
    return generated_images


def main():
    """Main inference function"""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("🚀 Starting conditional MAISI inference")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Load configuration
    config = load_config(args.config)

    # Load models
    unet, autoencoder, noise_scheduler, scale_factor = load_models(
        config, args.checkpoint, device, logger, args.use_fp32
    )

    # Determine what to generate
    if args.samples:
        # Generate specific numbers for each class
        sample_counts = [int(x.strip()) for x in args.samples.split(',')]
        if len(sample_counts) != 4:
            raise ValueError("--samples must contain exactly 4 comma-separated integers")

        logger.info(f"📊 Generation plan: {dict(zip(CLASS_NAMES, sample_counts))}")

        for class_idx, (class_name, num_samples) in enumerate(zip(CLASS_NAMES, sample_counts)):
            if num_samples > 0:
                generate_images_for_class(
                    unet, autoencoder, noise_scheduler, scale_factor, device,
                    class_idx, class_name, num_samples, args.output_dir, args, logger
                )

    elif args.class_name:
        # Generate for specific class
        class_idx = CLASS_MAPPING[args.class_name]
        generate_images_for_class(
            unet, autoencoder, noise_scheduler, scale_factor, device,
            class_idx, args.class_name, args.num_samples, args.output_dir, args, logger
        )

    else:
        # Default: generate 10 images for each class
        logger.info("📊 No specific generation plan provided, generating 10 images per class")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            generate_images_for_class(
                unet, autoencoder, noise_scheduler, scale_factor, device,
                class_idx, class_name, 10, args.output_dir, args, logger
            )

    logger.info("🎉 Inference completed successfully!")
    logger.info(f"Generated images saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
