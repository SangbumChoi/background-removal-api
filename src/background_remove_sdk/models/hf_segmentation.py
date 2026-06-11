"""Hugging Face transformers backends for BiRefNet-style checkpoints.

Covers BiRefNet and BRIA RMBG-2.0 (which is built on the BiRefNet
architecture): the variant is a Hugging Face repo id loaded with
``AutoModelForImageSegmentation`` + ``trust_remote_code``.

Native DTO: torch tensors — a normalized 1024x1024 batch in, sigmoid logits
out — translated here to the SDK's canonical PIL mask.
"""

from __future__ import annotations

from PIL import Image

from background_remove_sdk.models.base import BaseBackend, to_mask


class HFSegmentationBackend(BaseBackend):
    name = "hf-segmentation"
    image_size = 1024
    description = "BiRefNet-style Hugging Face image segmentation checkpoints."
    install_hint = 'pip install "background-remove-sdk[hf]"'

    def _load_model(self):
        import torch
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            self.variant, trust_remote_code=True
        )
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._torch_device = device
        return model

    def _preprocess(self, image: Image.Image):
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(image.convert("RGB")).unsqueeze(0)

    def predict_mask(self, image: Image.Image) -> Image.Image:
        model = self._get_model()
        import torch

        batch = self._preprocess(image).to(self._torch_device)
        with torch.no_grad():
            # BiRefNet-family models return multi-scale outputs; the last is final.
            logits = model(batch)[-1]
        return to_mask(logits.sigmoid()[0], size=image.size)


class BiRefNetBackend(HFSegmentationBackend):
    name = "birefnet"
    default_variant = "ZhengPeng7/BiRefNet"
    variants = (
        "ZhengPeng7/BiRefNet",
        "ZhengPeng7/BiRefNet_lite",
        "ZhengPeng7/BiRefNet-portrait",
        "ZhengPeng7/BiRefNet-HRSOD",
    )
    description = "BiRefNet (CAAI AIR 2024) via transformers. High-quality edges; needs GPU for speed."


class RMBGBackend(HFSegmentationBackend):
    name = "rmbg"
    default_variant = "briaai/RMBG-2.0"
    variants = ("briaai/RMBG-2.0",)
    description = (
        "BRIA RMBG-2.0 via transformers. Strong on complex backgrounds. "
        "Note: weights are licensed for non-commercial use. "
        "(For RMBG-1.4 use the rembg backend: rembg:bria-rmbg.)"
    )
