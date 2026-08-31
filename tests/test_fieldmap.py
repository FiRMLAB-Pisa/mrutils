"""Off-resonance from a multi-echo acquisition."""

import numpy as np
import pytest

import mrutils as mru


def _echoes(offset, echo_times, to_array):
    stack = np.stack([np.exp(2j * np.pi * offset * te) for te in echo_times])
    return to_array(stack)


def test_a_uniform_offset_is_recovered(to_array, as_numpy):
    offset = np.zeros((16, 16), dtype=np.float32)
    offset[:, 8:] = 40.0
    echo_times = [5e-3, 7e-3]
    measured = mru.field_map(
        _echoes(offset, echo_times, to_array), echo_times, smoothing=0.0
    )
    assert np.allclose(as_numpy(measured), offset, atol=1e-3)


def test_several_echoes_are_averaged_coherently(to_array, as_numpy):
    offset = np.full((8, 8), 25.0, dtype=np.float32)
    echo_times = [4e-3, 6e-3, 8e-3, 10e-3]
    measured = mru.field_map(
        _echoes(offset, echo_times, to_array), echo_times, smoothing=0.0
    )
    assert np.allclose(as_numpy(measured), offset, atol=1e-3)


def test_an_offset_beyond_the_spacing_comes_back_as_its_alias(to_array, as_numpy):
    # The spacing bounds what the map can state: +-1/(2 dTE).
    spacing = 2e-3
    limit = 1.0 / (2.0 * spacing)
    offset = np.full((4, 4), limit + 60.0, dtype=np.float32)
    echo_times = [5e-3, 5e-3 + spacing]
    measured = as_numpy(
        mru.field_map(_echoes(offset, echo_times, to_array), echo_times, smoothing=0.0)
    )
    assert np.allclose(measured, offset - 2.0 * limit, atol=1e-2)


def test_widening_the_spacing_narrows_the_range_it_can_state():
    narrow = [5e-3, 5.5e-3]
    wide = [5e-3, 9e-3]
    offset = np.full((4, 4), 300.0, dtype=np.float32)
    unaliased = mru.field_map(
        _echoes(offset, narrow, np.asarray), narrow, smoothing=0.0
    )
    aliased = mru.field_map(_echoes(offset, wide, np.asarray), wide, smoothing=0.0)
    assert np.allclose(unaliased, offset, atol=1e-2)
    assert not np.allclose(aliased, offset, atol=1.0)


def test_smoothing_the_product_survives_a_wrap(to_array, as_numpy):
    rng = np.random.default_rng(0)
    offset = np.full((16, 16), 30.0, dtype=np.float32)
    echo_times = [5e-3, 7e-3]
    stack = np.stack([np.exp(2j * np.pi * offset * te) for te in echo_times])
    noisy = stack + 0.05 * (
        rng.normal(size=stack.shape) + 1j * rng.normal(size=stack.shape)
    )
    measured = as_numpy(mru.field_map(to_array(noisy), echo_times, smoothing=2.0))
    assert np.abs(measured - offset).mean() < 2.0


def test_a_single_echo_is_refused():
    with pytest.raises(ValueError, match="at least two echoes"):
        mru.field_map(np.ones((1, 4, 4), dtype=complex), [5e-3])


def test_unevenly_spaced_echoes_are_refused():
    with pytest.raises(ValueError, match="evenly spaced"):
        mru.field_map(np.ones((3, 4, 4), dtype=complex), [1e-3, 2e-3, 5e-3])


def test_a_mismatched_number_of_echo_times_is_refused():
    with pytest.raises(ValueError, match="do not match"):
        mru.field_map(np.ones((2, 4, 4), dtype=complex), [1e-3, 2e-3, 3e-3])
