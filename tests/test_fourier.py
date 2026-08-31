"""The one Fourier convention, and centered resizing."""

import numpy as np
import pytest

import mrutils as mru


def test_the_round_trip_is_exact(to_array, as_numpy):
    rng = np.random.default_rng(0)
    image = to_array(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    recovered = mru.ifftc(mru.fftc(image))
    assert np.allclose(as_numpy(recovered), as_numpy(image), atol=1e-10)


def test_a_point_at_the_image_centre_is_flat_in_kspace(to_array, as_numpy):
    image = np.zeros((16, 16), dtype=complex)
    image[8, 8] = 1.0
    kspace = as_numpy(mru.fftc(to_array(image)))
    assert np.allclose(np.abs(kspace), np.abs(kspace).mean())


def test_the_transform_preserves_energy(to_array, as_numpy):
    rng = np.random.default_rng(1)
    image = rng.normal(size=(12, 10)) + 1j * rng.normal(size=(12, 10))
    kspace = as_numpy(mru.fftc(to_array(image)))
    assert np.isclose(np.linalg.norm(kspace), np.linalg.norm(image))


def test_the_dc_term_lands_at_the_grid_centre(to_array, as_numpy):
    ones = to_array(np.ones((16, 16), dtype=complex))
    kspace = as_numpy(mru.fftc(ones))
    assert np.argmax(np.abs(kspace)) == np.ravel_multi_index((8, 8), (16, 16))


def test_a_single_axis_transform_leaves_the_others_alone(to_array, as_numpy):
    rng = np.random.default_rng(2)
    volume = rng.normal(size=(4, 8, 8)) + 1j * rng.normal(size=(4, 8, 8))
    per_plane = np.stack(
        [as_numpy(mru.fftc(to_array(plane), axes=-1)) for plane in volume]
    )
    together = as_numpy(mru.fftc(to_array(volume), axes=-1))
    assert np.allclose(per_plane, together, atol=1e-10)


def test_resize_centered_pads_symmetrically_about_the_centre(to_array, as_numpy):
    value = np.arange(4.0).reshape(1, 4)
    padded = as_numpy(mru.resize_centered(to_array(value), (1, 8)))
    assert padded.shape == (1, 8)
    assert np.allclose(padded[0, 2:6], value[0])
    assert np.allclose(padded[0, :2], 0.0) and np.allclose(padded[0, 6:], 0.0)


def test_resize_centered_crops_back_to_what_it_padded(to_array, as_numpy):
    rng = np.random.default_rng(3)
    value = rng.normal(size=(2, 6))
    padded = mru.resize_centered(to_array(value), (2, 14))
    assert np.allclose(as_numpy(mru.resize_centered(padded, (2, 6))), value)


def test_resize_centered_axis_leaves_a_matching_axis_alone(to_array):
    value = to_array(np.zeros((3, 5)))
    assert mru.resize_centered_axis(value, 5, axis=-1) is value


def test_a_torch_tensor_keeps_its_device():
    torch = pytest.importorskip("torch")
    image = torch.zeros((8, 8), dtype=torch.complex64)
    assert mru.fftc(image).device == image.device
    assert mru.resize_centered(image, (16, 16)).device == image.device


def test_the_namespace_of_the_input_is_the_namespace_of_the_result(to_array):
    result = mru.fftc(to_array(np.zeros((4, 4), dtype=complex)))
    assert (
        type(result).__module__.split(".")[0]
        == type(to_array(np.zeros(1))).__module__.split(".")[0]
    )
