"""Core background removal API.

The heavy model (InSpyReNet, via the ``transparent-background`` package) is
loaded lazily on first use, so importing this module is cheap.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]

_DEFAULT_SUFFIXES = {
    "rgba": "_no_bg",
    "mask": "_mask",
    "object": "_object",
}


def load_image(image: ImageInput) -> Image.Image:
    """Coerce a path, raw bytes, numpy array, or PIL image into a PIL image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path)
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(
        f"Unsupported image input type: {type(image).__name__}. "
        "Expected a file path, bytes, numpy array, or PIL.Image."
    )


def default_output_path(image: ImageInput, kind: str = "rgba") -> Path:
    """Derive an output PNG path next to the input file when possible."""
    suffix = _DEFAULT_SUFFIXES.get(kind, "_out")
    if isinstance(image, (str, Path)):
        path = Path(image)
        return path.with_name(f"{path.stem}{suffix}.png")
    return Path(f"output{suffix}.png")


class BackgroundRemover:
    """Reusable, model-agnostic background remover.

    Instantiate once and call :meth:`remove` repeatedly to amortize model
    loading. All methods accept a file path, raw bytes, a numpy array, or a
    PIL image as input, and every model's native output is normalized to the
    same PIL types.

    Args:
        model: a ``"backend[:variant]"`` spec string, e.g. ``"inspyrenet"``,
            ``"inspyrenet:fast"``, ``"rembg:isnet-anime"``, ``"birefnet"``,
            ``"rmbg"``, ``"ben2"``. See
            :func:`background_remove_sdk.models.list_models`.
        device: device hint such as ``"cuda:0"`` or ``"cpu"``.
            Auto-detected when ``None``.
        **model_options: extra options forwarded to the backend.
    """

    def __init__(self, model: str = "inspyrenet", device: Optional[str] = None, **model_options):
        self.model = model
        self.device = device
        self.model_options = model_options
        self._backend = None
        self._lock = threading.Lock()

    def _get_backend(self):
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    from background_remove_sdk.models import create_backend

                    self._backend = create_backend(
                        self.model, device=self.device, **self.model_options
                    )
        return self._backend

    def remove(self, image: ImageInput, output_path: Optional[Union[str, Path]] = None) -> Image.Image:
        """Remove the background from a single image.

        Returns an RGBA image where background pixels are transparent. When
        ``output_path`` is given, the result is also saved there as PNG.
        """
        img = load_image(image).convert("RGB")
        output = self._get_backend().remove(img)
        if output_path is not None:
            output.save(output_path, "PNG")
        return output

    def mask(self, image: ImageInput, output_path: Optional[Union[str, Path]] = None) -> Image.Image:
        """Return the foreground mask as a grayscale (``L``) image."""
        img = load_image(image).convert("RGB")
        mask = self._get_backend().predict_mask(img)
        if output_path is not None:
            mask.save(output_path, "PNG")
        return mask

    def extract_object(
        self,
        image: ImageInput,
        x: int,
        y: int,
        output_path: Optional[Union[str, Path]] = None,
        min_area: int = 1000,
    ) -> Image.Image:
        """Extract the foreground object containing the point ``(x, y)``.

        Removes the background, finds the connected foreground region that
        contains the given pixel, and returns the result cropped to that
        region's bounding box.

        Raises:
            ValueError: if the point lies outside the image or no foreground
                object of at least ``min_area`` pixels contains it.
        """
        rgba = self.remove(image)
        if not (0 <= x < rgba.width and 0 <= y < rgba.height):
            raise ValueError(
                f"Point ({x}, {y}) is outside the image bounds {rgba.width}x{rgba.height}."
            )
        alpha = rgba.getchannel("A")
        if alpha.getpixel((x, y)) == 0:
            raise ValueError(f"No foreground object found at point ({x}, {y}).")
        bounding_box = _bounding_box_at_point(alpha, x, y, min_area=min_area)
        output = rgba.crop(bounding_box)
        if output_path is not None:
            output.save(output_path, "PNG")
        return output


def _bounding_box_at_point(mask: Image.Image, x: int, y: int, min_area: int = 1000) -> Tuple[int, int, int, int]:
    """Find the bounding box of the mask contour containing ``(x, y)``."""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            cx, cy, w, h = cv2.boundingRect(contour)
            return (cx, cy, cx + w, cy + h)
    raise ValueError(
        f"No foreground object with area >= {min_area} pixels contains point ({x}, {y})."
    )


_default_removers: dict = {}
_default_removers_lock = threading.Lock()


def _shared_remover(model: str, device: Optional[str]) -> BackgroundRemover:
    key = (model, device)
    if key not in _default_removers:
        with _default_removers_lock:
            if key not in _default_removers:
                _default_removers[key] = BackgroundRemover(model=model, device=device)
    return _default_removers[key]


def remove_background(
    image: ImageInput,
    output_path: Optional[Union[str, Path]] = None,
    model: str = "inspyrenet",
    device: Optional[str] = None,
) -> Image.Image:
    """Remove the background from a single image.

    This is the one-call entrypoint of the SDK. The underlying model is
    loaded once per ``(model, device)`` combination and reused across calls.
    """
    return _shared_remover(model, device).remove(image, output_path=output_path)


def generate_mask(
    image: ImageInput,
    output_path: Optional[Union[str, Path]] = None,
    model: str = "inspyrenet",
    device: Optional[str] = None,
) -> Image.Image:
    """Generate the foreground mask for a single image."""
    return _shared_remover(model, device).mask(image, output_path=output_path)


def extract_object_at_point(
    image: ImageInput,
    x: int,
    y: int,
    output_path: Optional[Union[str, Path]] = None,
    model: str = "inspyrenet",
    device: Optional[str] = None,
) -> Image.Image:
    """Extract the foreground object containing the point ``(x, y)``."""
    return _shared_remover(model, device).extract_object(image, x, y, output_path=output_path)
