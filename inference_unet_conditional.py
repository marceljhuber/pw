#!/usr/bin/env python3
"""
Conditional MAISI Diffusion Model Inference Script
Generates medical images conditioned on class labels: CNV, DME, DRUSEN, NORMAL

Usage:
    python inference_unet_conditional.py --config config.json --checkpoint model.pt --samples 10,5,8,12
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
import logging

# Import your custom modules
from networks.conditional_maisi_wrapper import ConditionalMAISIWrapper
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.utils import define_instance
from scripts.sample import ReconModel

# Class mapping
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_MAPPING = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}


def create_timestamped_output_dir(base_dir="./generated_images"):
    """Create a timestamped output directory for this inference run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(
        output_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Conditional MAISI Diffusion Model Inference"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file used during training",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="./generated_images",
        help="Base directory for generated images (timestamped subdirectory will be created)",
    )

    # Generation options
    parser.add_argument(
        "--samples",
        type=str,
        help="Number of samples per class as comma-separated values (e.g., '10,5,8,12' for CNV,DME,DRUSEN,NORMAL)",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        choices=CLASS_NAMES,
        help="Generate images for specific class only",
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Generate images without class conditioning (true unconditional)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate (used with --class_name)",
    )

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--use_classifier_free_guidance",
        action="store_true",
        help="Enable classifier-free guidance",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Merge configs with priority handling
    merged_config = {}
    for section in [
        "main",
        "model_config",
        "env_config",
        "vae_def",
        "conditional_config",
        "model",
    ]:
        if section in config:
            merged_config.update(config[section])

    return argparse.Namespace(**merged_config)


def safe_load_checkpoint(checkpoint_path, device, logger):
    """Safely load checkpoint with MONAI compatibility"""
    try:
        logger.info("Loading checkpoint (MONAI compatible mode)...")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        logger.info("✅ Checkpoint loaded successfully")
        return checkpoint
    except Exception as e:
        logger.warning(f"⚠️  Failed to load checkpoint: {e}")
        return {}


def force_float32_model(model, logger):
    """Aggressively convert all model parameters to float32"""
    logger.info("🔧 Converting model to float32...")

    # Convert the entire model to float32
    model = model.float()

    # Recursively convert all parameters and buffers
    def convert_module_to_float32(module):
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                param.data = param.data.float()
                if param.grad is not None:
                    param.grad = param.grad.float()

        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                buffer.data = buffer.data.float()

        # Recursively apply to all submodules
        for child in module.children():
            convert_module_to_float32(child)

    convert_module_to_float32(model)

    # Verify conversion
    param_count = sum(1 for p in model.parameters())
    float32_count = sum(1 for p in model.parameters() if p.dtype == torch.float32)
    logger.info(
        f"✅ Model conversion complete: {float32_count}/{param_count} parameters are float32"
    )

    return model


def load_models(config, checkpoint_path, device, logger):
    """Load the trained diffusion model and autoencoder with aggressive float32 conversion"""
    logger.info("Loading models with comprehensive float32 conversion...")

    # Get conditional config parameters
    conditional_config = getattr(config, "conditional_config", {})
    if hasattr(config, "conditional_config"):
        conditional_config = config.conditional_config
    else:
        conditional_config = {
            "num_classes": getattr(config, "num_classes", 4),
            "class_emb_dim": getattr(config, "class_emb_dim", 64),
            "conditioning_method": getattr(
                config, "conditioning_method", "input_concat"
            ),
        }

    # Load conditional diffusion UNet
    logger.info("Loading conditional MAISI UNet...")
    unet = ConditionalMAISIWrapper(
        config_args=config,
        num_classes=conditional_config.get("num_classes", 4),
        class_emb_dim=conditional_config.get("class_emb_dim", 64),
        conditioning_method=conditional_config.get(
            "conditioning_method", "input_concat"
        ),
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

    # Force UNet to float32
    unet = force_float32_model(unet, logger)
    unet.eval()

    # Load noise scheduler
    logger.info("Loading noise scheduler...")
    noise_scheduler = define_instance(config, "noise_scheduler")

    # Load autoencoder with forced float32 config
    logger.info("Loading autoencoder with float32 configuration...")

    # Get autoencoder config and force float32 settings
    if hasattr(config, "model") and "autoencoder" in config.model:
        autoencoder_config = config.model["autoencoder"].copy()
    else:
        autoencoder_config = {
            "spatial_dims": getattr(config, "spatial_dims", 2),
            "in_channels": getattr(config, "image_channels", 1),
            "out_channels": getattr(config, "image_channels", 1),
            "latent_channels": getattr(config, "latent_channels", 4),
            "num_channels": [64, 128, 256],
            "num_res_blocks": [2, 2, 2],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "num_splits": 1,
            "dim_split": 1,
        }

    # Force disable all float16 options
    autoencoder_config["norm_float16"] = False

    # Create autoencoder
    autoencoder = AutoencoderKlMaisi(**autoencoder_config).to(device)

    # Load autoencoder checkpoint
    if hasattr(config, "trained_autoencoder_path") and config.trained_autoencoder_path:
        try:
            vae_checkpoint = safe_load_checkpoint(
                config.trained_autoencoder_path, device, logger
            )
            if "autoencoder_state_dict" in vae_checkpoint:
                autoencoder.load_state_dict(vae_checkpoint["autoencoder_state_dict"])
                logger.info("✅ Autoencoder loaded from checkpoint")
        except Exception as e:
            logger.warning(f"⚠️  Could not load autoencoder checkpoint: {e}")

    # Force autoencoder to float32
    autoencoder = force_float32_model(autoencoder, logger)
    autoencoder.eval()

    # Get scale factor
    scale_factor = (
        checkpoint.get("scale_factor", 1.0) if "scale_factor" in checkpoint else 1.0
    )
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
):
    """Generate a batch of images with class conditioning - pure float32"""
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    # Handle batch size calculation for unconditional generation
    if class_labels is None:
        batch_size = 1  # Default batch size for unconditional generation
    elif isinstance(class_labels, (list, tuple)):
        batch_size = len(class_labels)
    else:
        batch_size = class_labels.shape[0]

    # Create random noise in float32
    noise = torch.randn(
        batch_size, *latent_shape[1:], device=device, dtype=torch.float32
    )

    # Convert class labels to tensor if needed (skip if None)
    if class_labels is not None and isinstance(class_labels, (list, tuple)):
        class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)

    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    # Denoising loop
    for t in tqdm(noise_scheduler.timesteps, desc="Denoising"):
        timestep_tensor = t.unsqueeze(0).expand(batch_size).to(device)

        with torch.no_grad():
            # Ensure noise is float32
            noise = noise.float()

            if use_cfg and guidance_scale > 1.0 and class_labels is not None:
                # Classifier-free guidance (only when class_labels is provided)
                noise_pred_cond = unet(
                    noise, timestep_tensor, class_labels=class_labels
                )
                noise_pred_uncond = unet(noise, timestep_tensor, class_labels=None)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # Pure unconditional generation
                noise_pred = unet(noise, timestep_tensor, class_labels=class_labels)

            # Ensure predictions are float32
            noise_pred = noise_pred.float()

        # Handle scheduler step output
        step_output = noise_scheduler.step(noise_pred, t, noise)

        if hasattr(step_output, "prev_sample"):
            noise = step_output.prev_sample.float()
        elif isinstance(step_output, tuple):
            noise = step_output[0].float()
        else:
            noise = step_output.float()

    # Decode latents to images - ensure everything is float32
    with torch.no_grad():
        # Ensure decode input is float32
        decode_input = (noise / scale_factor).float()
        synthetic_images = recon_model(decode_input)
        synthetic_images = synthetic_images.float()

    return synthetic_images


def generate_images_for_class(
    unet,
    autoencoder,
    noise_scheduler,
    scale_factor,
    device,
    class_idx,
    class_name,
    num_samples,
    output_dir,
    args,
    logger,
):
    """Generate images for a specific class"""
    logger.info(
        f"🎯 Generating {num_samples} images for class {class_idx} ({class_name})"
    )

    # Create class-specific output directory
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    # Define latent shape
    latent_shape = [1, 4, 64, 64]  # [batch_size, channels, H, W] for 2D MAISI

    # Generate images in batches
    batch_size = min(args.batch_size, num_samples)
    generated_images = []

    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)

        if class_idx is None:
            class_labels = None
        else:
            class_labels = [class_idx] * current_batch_size

        current_latent_shape = [current_batch_size] + latent_shape[1:]

        logger.info(
            f"Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}"
        )

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
            )

            # Process and save images
            for j in range(current_batch_size):
                img_idx = i + j

                # Denormalize from [-1,1] to [0,1]
                image = (synthetic_images[j] + 1) / 2.0
                image = torch.clamp(image, 0, 1)

                # Fix rotation - rotate 90 degrees counterclockwise (k=-1 for clockwise)
                image = torch.rot90(image, k=1, dims=[-2, -1])  # Rotate in H,W plane

                # Convert to uint8 range [0,255]
                image = (image * 255).type(torch.uint8)

                # Save image with simpler naming (no timestamp in filename)
                img_saver = SaveImage(
                    output_dir=class_output_dir,
                    output_postfix=f"{class_name}_{img_idx:04d}",
                    output_ext=".png",
                    separate_folder=False,
                    print_log=False,
                )
                img_saver(image)

                generated_images.append(image)

            logger.info(f"✅ Successfully generated batch {i//batch_size + 1}")

        except Exception as e:
            logger.error(f"❌ Error generating batch {i//batch_size + 1}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Continuing with next batch...")
            continue

    logger.info(
        f"✅ Generated {len(generated_images)} images for {class_name} in {class_output_dir}"
    )
    return generated_images


def save_generation_summary(output_dir, args, generation_plan, logger):
    """Save a summary of the generation parameters and results"""
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "checkpoint_file": args.checkpoint,
        "generation_parameters": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_classifier_free_guidance": args.use_classifier_free_guidance,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
        },
        "generation_plan": generation_plan,
        "output_directory": output_dir,
    }

    summary_file = os.path.join(output_dir, "generation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"📄 Generation summary saved to: {summary_file}")


def main():
    """Main inference function"""
    args = parse_arguments()

    # Create timestamped output directory for this run
    output_dir = create_timestamped_output_dir(args.base_output_dir)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(
        "🚀 Starting conditional MAISI inference with comprehensive float32 conversion"
    )
    logger.info(f"📁 Output directory for this run: {output_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")

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

    # Load models with comprehensive float32 conversion
    unet, autoencoder, noise_scheduler, scale_factor = load_models(
        config, args.checkpoint, device, logger
    )

    # Determine what to generate
    generation_plan = {}

    if args.samples:
        # Generate specific numbers for each class
        sample_counts = [int(x.strip()) for x in args.samples.split(",")]
        if len(sample_counts) != 4:
            raise ValueError(
                "--samples must contain exactly 4 comma-separated integers"
            )

        generation_plan = dict(zip(CLASS_NAMES, sample_counts))
        logger.info(f"📊 Generation plan: {generation_plan}")

        for class_idx, (class_name, num_samples) in enumerate(
            zip(CLASS_NAMES, sample_counts)
        ):
            if num_samples > 0:
                generate_images_for_class(
                    unet,
                    autoencoder,
                    noise_scheduler,
                    scale_factor,
                    device,
                    class_idx,
                    class_name,
                    num_samples,
                    output_dir,
                    args,
                    logger,
                )

    elif args.class_name:
        # Generate for specific class
        class_idx = CLASS_MAPPING[args.class_name]
        generation_plan = {args.class_name: args.num_samples}
        logger.info(f"📊 Generation plan: {generation_plan}")

        generate_images_for_class(
            unet,
            autoencoder,
            noise_scheduler,
            scale_factor,
            device,
            class_idx,
            args.class_name,
            args.num_samples,
            output_dir,
            args,
            logger,
        )

    else:

        # Default: generate 10 images for each class
        generation_plan = dict(zip(CLASS_NAMES, [10, 10, 10, 10]))
        logger.info(f"📊 Generation plan: {generation_plan}")

        # Unconditional mode
        if args.unconditional:
            generation_plan = {"UNCONDITIONAL": args.num_samples}
            logger.info(f"📊 Generation plan (unconditional): {generation_plan}")
            generate_images_for_class(
                unet,
                autoencoder,
                noise_scheduler,
                scale_factor,
                device,
                class_idx=None,  # no label
                class_name="UNCONDITIONAL",
                num_samples=args.num_samples,
                output_dir=output_dir,
                args=args,
                logger=logger,
            )

            save_generation_summary(output_dir, args, generation_plan, logger)

            logger.info("🎉 Unconditional inference completed successfully!")

            return  # exit early so it doesn’t fall into class loop

        # Normal default mode (10 per class)
        for class_idx, class_name in enumerate(CLASS_NAMES):
            generate_images_for_class(
                unet,
                autoencoder,
                noise_scheduler,
                scale_factor,
                device,
                class_idx,
                class_name,
                10,
                output_dir,
                args,
                logger,
            )

    # Save generation summary
    save_generation_summary(output_dir, args, generation_plan, logger)

    logger.info("🎉 Inference completed successfully!")
    logger.info(f"📁 All generated images and logs saved in: {output_dir}")


if __name__ == "__main__":
    main()
