import pytest

from background_remove_sdk import cli, core


@pytest.fixture(autouse=True)
def _patch_model(patched_shared_remover):
    yield


def test_default_output(sample_image, capsys):
    assert cli.main([str(sample_image)]) == 0
    expected = sample_image.with_name("sample_no_bg.png")
    assert expected.exists()
    assert capsys.readouterr().out.strip() == str(expected)


def test_explicit_output(sample_image, tmp_path):
    out = tmp_path / "cutout.png"
    assert cli.main([str(sample_image), "-o", str(out)]) == 0
    assert out.exists()


def test_mask_flag(sample_image):
    assert cli.main([str(sample_image), "--mask"]) == 0
    assert sample_image.with_name("sample_mask.png").exists()


def test_point_flag(sample_image, tmp_path):
    out = tmp_path / "obj.png"
    assert cli.main([str(sample_image), "--point", "30", "20", "-o", str(out)]) == 0
    from PIL import Image

    assert Image.open(out).size == (40, 30)


def test_mask_and_point_conflict(sample_image, capsys):
    assert cli.main([str(sample_image), "--mask", "--point", "1", "2"]) == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_missing_input(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.jpg")]) == 1
    assert "error:" in capsys.readouterr().err
