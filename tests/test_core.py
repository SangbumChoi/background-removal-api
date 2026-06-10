import numpy as np
import pytest
from PIL import Image

from background_remove_sdk import core


class TestLoadImage:
    def test_from_path(self, sample_image):
        assert core.load_image(sample_image).size == (100, 80)

    def test_from_str_path(self, sample_image):
        assert core.load_image(str(sample_image)).size == (100, 80)

    def test_from_bytes(self, sample_image):
        assert core.load_image(sample_image.read_bytes()).size == (100, 80)

    def test_from_pil(self):
        img = Image.new("RGB", (10, 10))
        assert core.load_image(img) is img

    def test_from_numpy(self):
        array = np.zeros((5, 7, 3), dtype=np.uint8)
        assert core.load_image(array).size == (7, 5)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            core.load_image(tmp_path / "nope.jpg")

    def test_bad_type(self):
        with pytest.raises(TypeError):
            core.load_image(12345)


class TestDefaultOutputPath:
    def test_rgba_suffix(self, tmp_path):
        path = core.default_output_path(tmp_path / "photo.jpg", "rgba")
        assert path == tmp_path / "photo_no_bg.png"

    def test_mask_suffix(self):
        assert core.default_output_path("a/b/photo.jpg", "mask").name == "photo_mask.png"

    def test_non_path_input(self):
        assert core.default_output_path(b"bytes", "rgba").name == "output_no_bg.png"


class TestBackgroundRemover:
    def test_remove_returns_rgba(self, fake_remover, sample_image):
        output = fake_remover.remove(sample_image)
        assert output.mode == "RGBA"
        assert output.size == (100, 80)
        alpha = np.asarray(output.getchannel("A"))
        assert alpha[20, 30] == 255  # inside the red rectangle
        assert alpha[70, 90] == 0  # background corner

    def test_remove_saves_output(self, fake_remover, sample_image, tmp_path):
        out = tmp_path / "out.png"
        fake_remover.remove(sample_image, output_path=out)
        assert Image.open(out).mode == "RGBA"

    def test_mask_is_grayscale(self, fake_remover, sample_image):
        mask = fake_remover.mask(sample_image)
        assert mask.mode == "L"
        assert mask.getpixel((30, 20)) == 255
        assert mask.getpixel((90, 70)) == 0

    def test_extract_object_crops_to_bbox(self, fake_remover, sample_image):
        output = fake_remover.extract_object(sample_image, x=30, y=20, min_area=100)
        # The red rectangle spans x 20..59, y 10..39 -> 40x30 crop.
        assert output.size == (40, 30)

    def test_extract_object_outside_image(self, fake_remover, sample_image):
        with pytest.raises(ValueError, match="outside the image"):
            fake_remover.extract_object(sample_image, x=500, y=500)

    def test_extract_object_on_background(self, fake_remover, sample_image):
        with pytest.raises(ValueError, match="No foreground object"):
            fake_remover.extract_object(sample_image, x=90, y=70)


class TestConvenienceFunctions:
    def test_remove_background(self, patched_shared_remover, sample_image, tmp_path):
        out = tmp_path / "result.png"
        output = core.remove_background(sample_image, output_path=out)
        assert output.mode == "RGBA"
        assert out.exists()

    def test_generate_mask(self, patched_shared_remover, sample_image):
        assert core.generate_mask(sample_image).mode == "L"

    def test_extract_object_at_point(self, patched_shared_remover, sample_image):
        output = core.extract_object_at_point(sample_image, 30, 20)
        assert output.size == (40, 30)
