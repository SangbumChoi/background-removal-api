import importlib.util

import numpy as np
import pytest
from PIL import Image

from background_remove_sdk import models
from background_remove_sdk.models import base
from conftest import FakeBackend


class TestParseModelSpec:
    def test_plain_backend(self):
        assert models.parse_model_spec("inspyrenet") == ("inspyrenet", None)

    def test_backend_with_variant(self):
        assert models.parse_model_spec("rembg:isnet-anime") == ("rembg", "isnet-anime")

    def test_variant_with_slash(self):
        name, variant = models.parse_model_spec("birefnet:ZhengPeng7/BiRefNet_lite")
        assert name == "birefnet"
        assert variant == "ZhengPeng7/BiRefNet_lite"

    def test_empty_spec_rejected(self):
        with pytest.raises(ValueError):
            models.parse_model_spec(":variant")


class TestRegistry:
    def test_builtin_backends_resolve(self):
        for name in ("inspyrenet", "rembg", "birefnet", "rmbg", "ben2"):
            cls = models.get_backend_class(name)
            assert issubclass(cls, base.BaseBackend)
            assert cls.default_variant

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown model backend"):
            models.get_backend_class("does-not-exist")

    def test_register_custom_backend(self):
        models.register_backend("fake-test", FakeBackend)
        try:
            backend = models.create_backend("fake-test:test")
            assert isinstance(backend, FakeBackend)
            assert backend.variant == "test"
        finally:
            models._registry.pop("fake-test", None)

    def test_register_rejects_non_backend(self):
        with pytest.raises(TypeError):
            models.register_backend("bad", dict)

    def test_list_models_is_light(self):
        info = models.list_models()
        assert "inspyrenet" in info and "rembg" in info
        for entry in info.values():
            assert entry["description"]
            assert entry["install"]

    def test_create_backend_passes_device(self):
        models.register_backend("fake-test", FakeBackend)
        try:
            backend = models.create_backend("fake-test", device="cuda:1")
            assert backend.device == "cuda:1"
        finally:
            models._registry.pop("fake-test", None)


class TestDTONormalization:
    def test_to_mask_from_pil_rgb(self):
        mask = base.to_mask(Image.new("RGB", (4, 4), (255, 255, 255)))
        assert mask.mode == "L" and mask.getpixel((0, 0)) == 255

    def test_to_mask_from_float_array(self):
        array = np.array([[0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
        mask = base.to_mask(array)
        assert mask.mode == "L"
        assert mask.getpixel((0, 0)) == 0
        assert mask.getpixel((1, 0)) == 255

    def test_to_mask_from_uint8_array(self):
        array = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        assert base.to_mask(array).getpixel((1, 0)) == 255

    def test_to_mask_squeezes_batch_dims(self):
        array = np.zeros((1, 1, 4, 4), dtype=np.float32)
        assert base.to_mask(array).size == (4, 4)

    def test_to_mask_resizes(self):
        array = np.ones((4, 4), dtype=np.float32)
        assert base.to_mask(array, size=(8, 6)).size == (8, 6)

    def test_to_mask_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="2D mask"):
            base.to_mask(np.zeros((2, 3, 4, 5, 6)))

    def test_compose_rgba(self):
        image = Image.new("RGB", (2, 2), (10, 20, 30))
        mask = Image.new("L", (2, 2), 255)
        rgba = base.compose_rgba(image, mask)
        assert rgba.mode == "RGBA"
        assert rgba.getpixel((0, 0)) == (10, 20, 30, 255)

    def test_compose_rgba_resizes_mask(self):
        image = Image.new("RGB", (4, 4))
        mask = Image.new("L", (2, 2), 255)
        assert base.compose_rgba(image, mask).size == (4, 4)

    def test_default_remove_composes_mask(self):
        backend = FakeBackend()
        image = Image.new("RGB", (4, 4), (255, 0, 0))
        rgba = backend.remove(image)
        assert rgba.mode == "RGBA"
        assert rgba.getpixel((0, 0))[3] == 255


@pytest.mark.skipif(
    importlib.util.find_spec("transparent_background") is not None,
    reason="transparent-background installed; missing-dependency path not reachable",
)
def test_missing_dependency_error_is_helpful():
    backend = models.create_backend("inspyrenet")
    with pytest.raises(base.ModelNotInstalledError, match="inspyrenet.*Install"):
        backend.predict_mask(Image.new("RGB", (4, 4)))
