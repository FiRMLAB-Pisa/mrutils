"""Receive-array compression and noise decorrelation."""

import numpy as np
import pytest

import mrutils as mru


def _redundant_lines(rng, independent=4, copies=2, samples=256):
    """An array of `independent * copies` channels carrying `independent` signals."""
    signals = rng.normal(size=(independent, samples)) + 1j * rng.normal(
        size=(independent, samples)
    )
    return np.concatenate([signals] * copies, axis=0)


def test_redundant_channels_compress_without_losing_the_signal(to_array, as_numpy):
    rng = np.random.default_rng(0)
    lines = _redundant_lines(rng)
    compressed, matrix = mru.coil_compress(to_array(lines), 4)
    assert as_numpy(compressed).shape == (4, 256)
    assert as_numpy(matrix).shape == (4, 8)
    # Eight channels carrying four signals live in a four-dimensional subspace,
    # so projecting onto it and back is the identity on the data.
    restored = as_numpy(matrix).conj().T @ as_numpy(compressed)
    assert np.allclose(restored, lines, atol=1e-8)


def test_an_energy_fraction_chooses_the_channel_count(to_array, as_numpy):
    rng = np.random.default_rng(1)
    lines = _redundant_lines(rng)
    compressed, _ = mru.coil_compress(to_array(lines), 0.999)
    assert as_numpy(compressed).shape[0] == 4


def test_a_count_beyond_the_physical_channels_keeps_them_all(to_array, as_numpy):
    rng = np.random.default_rng(2)
    lines = _redundant_lines(rng)
    compressed, _ = mru.coil_compress(to_array(lines), 99)
    assert as_numpy(compressed).shape[0] == 8


def test_the_basis_compresses_data_the_calibration_never_saw(to_array, as_numpy):
    rng = np.random.default_rng(3)
    lines = _redundant_lines(rng)
    _, matrix = mru.coil_compress(to_array(lines), 4)
    later = _redundant_lines(rng, samples=64)
    projected = as_numpy(matrix) @ later
    assert projected.shape == (4, 64)
    restored = as_numpy(matrix).conj().T @ projected
    assert np.allclose(restored, later, atol=1e-8)


def test_a_calibration_radius_without_a_trajectory_is_refused():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="given together"):
        mru.coil_compress(_redundant_lines(rng), 4, calibration_radius=0.1)


def test_a_measurement_that_is_not_coils_by_samples_is_refused():
    with pytest.raises(ValueError, match=r"\(coils, samples\)"):
        mru.coil_compress(np.zeros((2, 4, 8), dtype=complex), 2)


def test_keeping_no_channel_is_refused():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="at least one channel"):
        mru.coil_compress(_redundant_lines(rng), 0)


def test_prewhitening_the_noise_against_itself_gives_unit_covariance(
    to_array, as_numpy
):
    rng = np.random.default_rng(6)
    mixing = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    noise = mixing @ (
        rng.normal(size=(4, 8192)) + 1j * rng.normal(size=(4, 8192))
    ).astype(complex)
    whitened = as_numpy(
        mru.noise_prewhiten(to_array(noise), to_array(noise), coil_axis=0)
    )
    covariance = whitened @ whitened.conj().T / whitened.shape[-1]
    assert np.allclose(covariance, np.eye(4), atol=0.1)


def test_prewhitening_preserves_the_measurement_shape(to_array, as_numpy):
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(4, 512)) + 1j * rng.normal(size=(4, 512))
    kspace = rng.normal(size=(4, 64)) + 1j * rng.normal(size=(4, 64))
    whitened = mru.noise_prewhiten(to_array(kspace), to_array(noise), coil_axis=0)
    assert as_numpy(whitened).shape == (4, 64)


def test_mixing_array_libraries_is_refused():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    noise = rng.normal(size=(4, 128)) + 1j * rng.normal(size=(4, 128))
    with pytest.raises(TypeError, match="same array library"):
        mru.noise_prewhiten(torch.as_tensor(noise), noise, coil_axis=0)


def test_the_basis_is_a_materialized_array_not_a_lazy_conjugate_view():
    """A Torch conj() view refuses .numpy(); the basis is handed out to apply."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    lines = torch.as_tensor(_redundant_lines(rng))
    _, matrix = mru.coil_compress(lines, 4)
    assert not matrix.is_conj()
    assert matrix.numpy().shape == (4, 8)
