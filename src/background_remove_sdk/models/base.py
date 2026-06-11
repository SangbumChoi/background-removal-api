"""Universal backend interface.

Every model backend, whatever its native library speaks (PIL images, numpy
arrays, torch tensors, raw bytes), adapts to one canonical contract:

- input:  ``PIL.Image`` in RGB
- mask:   ``PIL.Image`` mode ``L``, 255 = foreground, same size as the input
- cutout: ``PIL.Image`` mode ``RGBA``

A backend only has to implement :meth:`BaseBackend.predict_mask`; the RGBA
cutout is composed from it by default. Backends whose library produces a
higher-quality cutout directly (e.g. with alpha matting) can override
:meth:`BaseBackend.remove`.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image


class ModelNotInstalledError(ImportError):
    """Raised when a backend's underlying package is not installed."""


class BaseBackend:
    """Base class for model backends.

    Args:
        variant: model variant or checkpoint identifier, backend-specific
            (e.g. ``"fast"`` for inspyrenet, ``"isnet-anime"`` for rembg,
            a Hugging Face repo id for birefnet/rmbg).
        device: device hint such as ``"cuda:0"`` or ``"cpu"``; backends that
            cannot honor it may ignore it.
        **options: extra backend-specific options.
    """

    #: registry name, e.g. ``"inspyrenet"``
    name: str = "base"
    #: variant used when none is given
    default_variant: Optional[str] = None
    #: known variants (informational; backends may accept others)
    variants: Tuple[str, ...] = ()
    #: short human-readable description for list_models()
    description: str = ""
    #: pip hint shown when the underlying package is missing
    install_hint: str = ""

    def __init__(self, variant: Optional[str] = None, device: Optional[str] = None, **options: Any):
        self.variant = variant or self.default_variant
        self.device = device
        self.options = options
        self._model = None
        self._lock = threading.Lock()

    # -- to be provided by subclasses ------------------------------------

    def _load_model(self):
        """Import the underlying library and return a loaded model/session."""
        raise NotImplementedError

    def predict_mask(self, image: Image.Image) -> Image.Image:
        """Return the foreground mask (mode ``L``) for an RGB PIL image."""
        raise NotImplementedError

    # -- shared machinery -------------------------------------------------

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = self._load_model()
                    except ImportError as exc:
                        raise ModelNotInstalledError(
                            f"The '{self.name}' backend requires an extra package. "
                            f"Install it with: {self.install_hint or 'see the README'}"
                        ) from exc
        return self._model

    def remove(self, image: Image.Image) -> Image.Image:
        """Return the RGBA cutout. Default: compose input + predicted mask."""
        return compose_rgba(image, self.predict_mask(image))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(variant={self.variant!r}, device={self.device!r})"


def to_mask(output: Any, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Normalize a model's native mask output to a PIL ``L`` image.

    Accepts PIL images, numpy arrays (float in [0, 1] or uint8), and torch
    tensors (detached and moved to CPU automatically). When ``size`` is
    given, the mask is resized to it (PIL ``(width, height)`` order).
    """
    if type(output).__module__.startswith("torch"):
        output = output.detach().cpu().numpy()
    if isinstance(output, Image.Image):
        mask = output.convert("L")
    else:
        array = np.asarray(output).squeeze()
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {array.shape}")
        if array.dtype != np.uint8:
            array = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
        mask = Image.fromarray(array, mode="L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.Resampling.BILINEAR)
    return mask


def compose_rgba(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply a foreground mask to an image as its alpha channel."""
    rgba = image.convert("RGBA")
    if mask.size != rgba.size:
        mask = mask.resize(rgba.size, Image.Resampling.BILINEAR)
    rgba.putalpha(mask)
    return rgba
