import os
from typing import Dict

import torch
import torchvision

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert
from torch_fidelity.interpolate_compat_tensorflow import (
    interpolate_bilinear_2d_like_tensorflow1x,
)
from torch_fidelity.registry import register_feature_extractor

from scripts.retfound_models_vit import RETFound_mae


def text_to_dtype(name, default="float32"):
    if name is None:
        name = default
    name = str(name).lower()
    if name == "float64":
        return torch.float64
    return torch.float32


def _extract_state_dict(ckpt: Dict) -> Dict:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict", "encoder"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_prefix(state: Dict, prefix: str) -> Dict:
    if not prefix:
        return state
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


class FeatureExtractorRETFoundMae(FeatureExtractorBase):
    INPUT_IMAGE_SIZE = 224

    def __init__(
        self,
        name,
        features_list,
        feature_extractor_weights_path=None,
        feature_extractor_internal_dtype=None,
        **kwargs,
    ):
        super().__init__(name, features_list)
        vassert(
            feature_extractor_weights_path is not None,
            "RETFound feature extractor requires feature_extractor_weights_path",
        )

        self.feature_extractor_internal_dtype = text_to_dtype(
            feature_extractor_internal_dtype, "float32"
        )

        vassert(
            os.path.exists(feature_extractor_weights_path),
            f"Weights not found: {feature_extractor_weights_path}",
        )

        self.model = RETFound_mae(global_pool=False, num_classes=0)

        ckpt = torch.load(feature_extractor_weights_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_prefix(state, "module.")
        state = _strip_prefix(state, "backbone.")
        state = _strip_prefix(state, "encoder.")
        state = _strip_prefix(state, "model.")

        self.model.load_state_dict(state, strict=False)

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        vassert(
            torch.is_tensor(x) and x.dtype == torch.uint8,
            "Expecting image as torch.Tensor with dtype=torch.uint8",
        )
        vassert(x.dim() == 4, f"Input is not BxCxHxW: {x.shape}")

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        vassert(x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")

        x = x.to(self.feature_extractor_internal_dtype)

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x, size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), align_corners=False
        )

        x = torchvision.transforms.functional.normalize(
            x,
            (255 * 0.485, 255 * 0.456, 255 * 0.406),
            (255 * 0.229, 255 * 0.224, 255 * 0.225),
            inplace=False,
        )

        feats = self.model.forward_features(x)
        if feats.dim() > 2:
            feats = feats.flatten(1)

        out = {"retfound": feats.to(torch.float32)}
        return tuple(out[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return ("retfound",)

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": "retfound",
            "fid": "retfound",
            "kid": "retfound",
            "prc": "retfound",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)


def register_retfound_feature_extractor():
    register_feature_extractor("retfound-mae", FeatureExtractorRETFoundMae)
