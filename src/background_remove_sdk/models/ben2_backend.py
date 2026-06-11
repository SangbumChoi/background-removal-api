"""BEN2 (Background Erase Network 2) backend.

Native DTO: PIL image in, foreground RGBA PIL image out via
``model.inference()``. The ``ben2`` package is not on PyPI; it installs from
GitHub.
"""

from __future__ import annotations

from PIL import Image

from background_remove_sdk.models.base import BaseBackend


class BEN2Backend(BaseBackend):
    name = "ben2"
    default_variant = "PramaLLC/BEN2"
    variants = ("PramaLLC/BEN2",)
    description = "BEN2 with Confidence Guided Matting. Excellent hair/edge matting."
    install_hint = "pip install git+https://github.com/PramaLLC/BEN2.git"

    def _load_model(self):
        import torch
        from ben2 import BEN_Base

        model = BEN_Base.from_pretrained(self.variant)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model

    def remove(self, image: Image.Image) -> Image.Image:
        output = self._get_model().inference(image.convert("RGB"))
        return output.convert("RGBA")

    def predict_mask(self, image: Image.Image) -> Image.Image:
        # BEN2 produces the matted foreground directly; the mask is its alpha.
        return self.remove(image).getchannel("A")
