import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from background_remove_sdk.core import BackgroundRemover  # noqa: E402


class FakeInSpyReNet:
    """Stand-in for transparent_background.Remover.

    Treats every pixel with red channel > 127 as foreground.
    """

    def process(self, image, type="rgba"):
        rgb = np.asarray(image.convert("RGB"))
        foreground = rgb[:, :, 0] > 127
        if type == "map":
            return Image.fromarray((foreground * 255).astype(np.uint8), mode="L")
        rgba = np.dstack([rgb, (foreground * 255).astype(np.uint8)])
        return Image.fromarray(rgba, mode="RGBA")


@pytest.fixture
def fake_remover():
    remover = BackgroundRemover()
    remover._remover = FakeInSpyReNet()
    return remover


@pytest.fixture
def sample_image(tmp_path):
    """100x80 image: red 40x30 rectangle at (20, 10) on a black background."""
    array = np.zeros((80, 100, 3), dtype=np.uint8)
    array[10:40, 20:60] = [255, 0, 0]
    path = tmp_path / "sample.jpg"
    Image.fromarray(array).save(path, quality=100, subsampling=0)
    return path


@pytest.fixture
def patched_shared_remover(monkeypatch, fake_remover):
    """Route the module-level convenience functions through the fake model."""
    from background_remove_sdk import core

    monkeypatch.setattr(core, "_shared_remover", lambda mode, device: fake_remover)
    return fake_remover
