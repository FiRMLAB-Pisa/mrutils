"""Receive-field shading correction."""

import numpy as np
import pytest

import mrutils as mru

pytest.importorskip("SimpleITK", reason="bias field correction needs the bias extra")


@pytest.fixture
def shaded():
    """A uniform block seen through a left-to-right shading."""
    truth = np.zeros((48, 48), dtype=np.float32)
    truth[10:38, 10:38] = 1.0
    field = np.linspace(0.5, 1.5, 48, dtype=np.float32)[None].repeat(48, axis=0)
    return truth, field, truth * field


def test_a_smooth_shading_is_divided_out(shaded):
    truth, _, observed = shaded
    corrected = mru.bias_field_correct(observed)
    inside = truth > 0

    def unevenness(image):
        values = image[inside]
        return float(values.std() / values.mean())

    assert unevenness(corrected) < 0.5 * unevenness(observed)


def test_the_field_comes_back_when_asked(shaded):
    _, _, observed = shaded
    corrected, field = mru.bias_field_correct(observed, return_field=True)
    assert corrected.shape == field.shape == observed.shape
    assert np.all(field > 0)


def test_a_torch_image_stays_a_torch_tensor_on_its_device(shaded):
    torch = pytest.importorskip("torch")
    _, _, observed = shaded
    tensor = torch.as_tensor(observed)
    corrected = mru.bias_field_correct(tensor)
    assert isinstance(corrected, torch.Tensor)
    assert corrected.device == tensor.device
    assert corrected.dtype == tensor.dtype


def test_a_supplied_mask_is_used(shaded):
    truth, _, observed = shaded
    corrected = mru.bias_field_correct(observed, mask=(truth > 0).astype(np.uint8))
    assert corrected.shape == observed.shape


def test_a_volume_is_corrected_as_readily_as_a_slice():
    volume = np.zeros((16, 24, 24), dtype=np.float32)
    volume[4:12, 6:18, 6:18] = 1.0
    field = np.linspace(0.6, 1.4, 24, dtype=np.float32)[None, None]
    assert mru.bias_field_correct(volume * field).shape == volume.shape


def test_a_four_dimensional_image_is_refused():
    with pytest.raises(ValueError, match="2D or 3D"):
        mru.bias_field_correct(np.ones((2, 4, 8, 8), dtype=np.float32))


def test_a_non_positive_shrink_factor_is_refused():
    with pytest.raises(ValueError, match="shrink_factor must be positive"):
        mru.bias_field_correct(np.ones((8, 8), dtype=np.float32), shrink_factor=0)
