"""InSpyReNet backend via the ``transparent-background`` package.

Native DTO: PIL image in, PIL image (RGBA or grayscale map) out.
"""

from __future__ import annotations

from PIL import Image

from background_remove_sdk.models.base import BaseBackend, to_mask


class InSpyReNetBackend(BaseBackend):
    name = "inspyrenet"
    default_variant = "base"
    variants = ("base", "fast", "base-nightly")
    description = "InSpyReNet salient object detection (transparent-background). Good default."
    install_hint = 'pip install "background-remove-sdk"  (installed by default)'

    def _load_model(self):
        from transparent_background import Remover

        kwargs = {"mode": self.variant, "jit": self.options.get("jit", False)}
        if self.device is not None:
            kwargs["device"] = self.device
        return Remover(**kwargs)

    def predict_mask(self, image: Image.Image) -> Image.Image:
        output = self._get_model().process(image, type="map")
        return to_mask(output, size=image.size)

    def remove(self, image: Image.Image) -> Image.Image:
        # The library composes the RGBA cutout itself; use it directly.
        output = self._get_model().process(image, type="rgba")
        if not isinstance(output, Image.Image):
            output = Image.fromarray(output)
        return output.convert("RGBA")
