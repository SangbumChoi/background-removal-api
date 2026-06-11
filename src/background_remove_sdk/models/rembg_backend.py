"""rembg backend — one adapter for the whole rembg model zoo (ONNX runtime).

Native DTO: mirrors its input type (PIL in -> PIL out, bytes in -> bytes out).
The variant is any rembg model name; the ONNX weights are downloaded on
first use.
"""

from __future__ import annotations

from PIL import Image

from background_remove_sdk.models.base import BaseBackend, to_mask


class RembgBackend(BaseBackend):
    name = "rembg"
    default_variant = "u2net"
    variants = (
        "u2net",
        "u2netp",
        "u2net_human_seg",
        "u2net_cloth_seg",
        "silueta",
        "isnet-general-use",
        "isnet-anime",
        "birefnet-general",
        "birefnet-general-lite",
        "birefnet-portrait",
        "birefnet-dis",
        "birefnet-hrsod",
        "birefnet-cod",
        "birefnet-massive",
        "bria-rmbg",
        "sam",
    )
    description = "rembg model zoo (U2-Net, IS-Net, BiRefNet, BRIA RMBG, ...) on ONNX runtime."
    install_hint = 'pip install "background-remove-sdk[rembg]" (or [rembg-gpu] for CUDA)'

    def _load_model(self):
        from rembg import new_session

        return new_session(self.variant)

    def predict_mask(self, image: Image.Image) -> Image.Image:
        session = self._get_model()  # raises a helpful error if rembg is missing
        from rembg import remove

        output = remove(image, session=session, only_mask=True)
        return to_mask(output, size=image.size)

    def remove(self, image: Image.Image) -> Image.Image:
        session = self._get_model()
        from rembg import remove

        output = remove(image, session=session, **self.options)
        return output.convert("RGBA")
