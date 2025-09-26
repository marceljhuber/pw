"""
Conditional MAISI Wrapper for Class-Conditioned Diffusion Training
"""

import torch
import torch.nn as nn
from scripts.utils import define_instance


class ConditionalMAISIWrapper(nn.Module):
    """
    Wrapper around MAISI DiffusionModelUNetMaisi to add class conditioning
    for training conditional diffusion models on medical imaging data.

    Supports CNV, DME, DRUSEN, NORMAL classes (indices 0-3).
    """

    def __init__(self, config_args, num_classes=4, class_emb_dim=64, conditioning_method='input_concat'):
        """
        Args:
            config_args: Configuration arguments containing diffusion_unet_def
            num_classes: Number of classes (default: 4 for CNV, DME, DRUSEN, NORMAL)
            class_emb_dim: Dimension of class embeddings
            conditioning_method: 'time_embedding' or 'input_concat'
        """
        super().__init__()

        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.conditioning_method = conditioning_method

        # Create the original MAISI UNet
        self.base_unet = define_instance(config_args, "diffusion_unet_def")

        # Add class embedding layer (+1 for unconditional)
        self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)

        if conditioning_method == 'time_embedding':
            self._setup_time_embedding_conditioning()
        elif conditioning_method == 'input_concat':
            self._setup_input_concatenation()
        else:
            raise ValueError(f"Unknown conditioning method: {conditioning_method}")

    def _setup_time_embedding_conditioning(self):
        """Setup class conditioning through time embeddings"""
        # Get the time embedding dimension from MAISI UNet
        if hasattr(self.base_unet, 'time_embed'):
            # From the source code, time_embed_dim = num_channels[0] * 4
            time_emb_dim = self.base_unet.block_out_channels[0] * 4
            self.class_proj = nn.Linear(self.class_emb_dim, time_emb_dim)
            print("⚙️  Using time embedding conditioning method")
        else:
            raise ValueError("Could not find time embedding in MAISI UNet")

    def _setup_input_concatenation(self):
        """Setup class conditioning through input concatenation"""
        print("🔧 Setting up input concatenation conditioning...")

        # Access the conv_in layer from MAISI UNet
        if not hasattr(self.base_unet, 'conv_in'):
            raise ValueError("MAISI UNet does not have conv_in layer")

        original_conv_block = self.base_unet.conv_in
        print(f"🎯 Found conv_in layer: {type(original_conv_block)}")

        # Extract the actual conv layer from MONAI Convolution block
        actual_conv = self._extract_conv_from_monai_block(original_conv_block)
        if actual_conv is None:
            raise ValueError("Could not extract conv layer from MONAI Convolution block")

        original_in_channels = actual_conv.in_channels
        new_in_channels = original_in_channels + self.class_emb_dim

        print(f"📊 Original input channels: {original_in_channels}")
        print(f"🔢 Adding class embedding channels: {self.class_emb_dim}")
        print(f"🎯 New input channels: {new_in_channels}")

        # Create new conv layer with additional channels
        new_conv = self._create_new_conv_layer(actual_conv, original_in_channels)

        # Create new MONAI Convolution block with the modified conv layer
        from monai.networks.blocks import Convolution

        new_conv_block = Convolution(
            spatial_dims=3,  # MAISI uses 3D
            in_channels=new_in_channels,
            out_channels=actual_conv.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # Replace the actual conv layer in the new block
        self._replace_conv_in_monai_block(new_conv_block, new_conv)

        # Replace the original conv_in with our new block
        self.base_unet.conv_in = new_conv_block
        # Also update the in_channels attribute
        self.base_unet.in_channels = new_in_channels

        self.use_input_conditioning = True
        print("✅ Input concatenation setup complete")

    def _extract_conv_from_monai_block(self, monai_conv_block):
        """Extract the actual PyTorch Conv layer from MONAI Convolution block"""
        # MONAI Convolution is a Sequential block
        if hasattr(monai_conv_block, 'conv'):
            return monai_conv_block.conv

        # Search through the Sequential modules
        for module in monai_conv_block:
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                return module

        return None

    def _replace_conv_in_monai_block(self, monai_conv_block, new_conv):
        """Replace the conv layer inside MONAI Convolution block"""
        # Find and replace the conv layer
        for i, module in enumerate(monai_conv_block):
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                monai_conv_block[i] = new_conv
                break

    def _create_new_conv_layer(self, original_conv, original_in_channels):
        """Create new conv layer with additional input channels"""
        new_in_channels = original_in_channels + self.class_emb_dim

        # Create new conv layer (MAISI uses 3D convolutions)
        if isinstance(original_conv, nn.Conv3d):
            new_conv = nn.Conv3d(
                new_in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None
            )
        elif isinstance(original_conv, nn.Conv2d):
            new_conv = nn.Conv2d(
                new_in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None
            )
        else:
            raise ValueError(f"Unsupported conv layer type: {type(original_conv)}")

        # Initialize weights properly
        with torch.no_grad():
            # Copy original weights
            new_conv.weight[:, :original_in_channels] = original_conv.weight
            # Initialize new channels to 0
            new_conv.weight[:, original_in_channels:] = 0

            # Copy bias if it exists - use nn.Parameter for proper assignment
            if new_conv.bias is not None and original_conv.bias is not None:
                new_conv.bias = nn.Parameter(original_conv.bias.clone())

        return new_conv

    def forward(self, x, timesteps, class_labels=None, **kwargs):
        """
        Forward pass with optional class conditioning

        Args:
            x: Input tensor [batch_size, channels, H, W, D] or [batch_size, channels, H, W]
            timesteps: Timestep tensor
            class_labels: Class labels tensor [batch_size] (0-3 for classes, None for unconditional)
            **kwargs: Additional arguments passed to base UNet
        """
        batch_size = x.shape[0]

        # Handle unconditional generation (for classifier-free guidance)
        if class_labels is None:
            class_labels = torch.full((batch_size,), self.num_classes,
                                    device=x.device, dtype=torch.long)

        # Get class embeddings
        class_emb = self.class_embedding(class_labels)  # [batch_size, class_emb_dim]

        if self.conditioning_method == 'time_embedding':
            return self._forward_time_embedding(x, timesteps, class_emb, **kwargs)
        else:
            return self._forward_input_concat(x, timesteps, class_emb, **kwargs)

    def _forward_time_embedding(self, x, timesteps, class_emb, **kwargs):
        """Forward pass using time embedding conditioning"""
        # Project class embedding and add to time embedding
        class_emb_proj = self.class_proj(class_emb)

        # We need to modify the internal time embedding
        # For now, we'll pass it through normally and let the base UNet handle it
        # A full implementation would require modifying the MAISI UNet's forward method
        return self.base_unet(x, timesteps, **kwargs)

    def _forward_input_concat(self, x, timesteps, class_emb, **kwargs):
        """Forward pass using input concatenation conditioning"""
        # Expand class embedding to match spatial dimensions
        spatial_dims = x.shape[2:]  # Get H, W, (D)
        class_emb_expanded = class_emb.view(x.shape[0], -1, *[1] * len(spatial_dims))
        class_emb_expanded = class_emb_expanded.expand(-1, -1, *spatial_dims)

        # Concatenate with input
        x_conditioned = torch.cat([x, class_emb_expanded], dim=1)

        return self.base_unet(x_conditioned, timesteps, **kwargs)

    def load_pretrained_base_weights(self, checkpoint_path, strict=False):
        """
        Load pretrained weights for the base MAISI UNet

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce state dict keys
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract base UNet state dict
        if 'unet_state_dict' in checkpoint:
            base_state_dict = {}
            for key, value in checkpoint['unet_state_dict'].items():
                if not key.startswith('class_embedding') and not key.startswith('class_proj'):
                    base_state_dict[key] = value

            # Try to load, ignoring conv_in layer if it has different input channels
            missing_keys, unexpected_keys = self.base_unet.load_state_dict(base_state_dict, strict=False)

            if missing_keys or unexpected_keys:
                print(f"⚠️  Loading with missing keys: {missing_keys}")
                print(f"⚠️  Loading with unexpected keys: {unexpected_keys}")

            print(f"✅ Loaded pretrained base UNet weights from {checkpoint_path}")
        else:
            print(f"⚠️  Warning: No 'unet_state_dict' found in {checkpoint_path}")


def create_conditional_maisi_unet(config_args, num_classes=4, class_emb_dim=64,
                                conditioning_method='input_concat'):
    """
    Factory function to create a conditional MAISI UNet

    Args:
        config_args: Configuration arguments
        num_classes: Number of classes (CNV=0, DME=1, DRUSEN=2, NORMAL=3)
        class_emb_dim: Class embedding dimension
        conditioning_method: 'time_embedding' or 'input_concat'

    Returns:
        ConditionalMAISIWrapper instance
    """
    return ConditionalMAISIWrapper(
        config_args=config_args,
        num_classes=num_classes,
        class_emb_dim=class_emb_dim,
        conditioning_method=conditioning_method
    )
