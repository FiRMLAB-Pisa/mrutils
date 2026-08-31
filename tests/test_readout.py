"""Readout-oversampling removal."""

import numpy as np
import pytest

import mrutils as mru


def test_removing_oversampling_halves_the_readout(to_array, as_numpy):
    rng = np.random.default_rng(0)
    readout = mru.fftc(
        to_array(rng.normal(size=(4, 128)) + 1j * rng.normal(size=(4, 128))), axes=-1
    )
    assert as_numpy(mru.remove_readout_oversampling(readout, 64)).shape == (4, 64)


def test_removing_oversampling_keeps_the_object_it_cropped_around(to_array, as_numpy):
    # An object confined to the middle half of the field of view is exactly
    # what twofold oversampling leaves room for, so the crop is lossless.
    image = np.zeros((1, 128), dtype=complex)
    image[0, 32:96] = np.linspace(0.2, 1.0, 64)
    kspace = mru.fftc(to_array(image), axes=-1)
    cropped = mru.remove_readout_oversampling(kspace, 64)
    # Both transforms are orthonormal, so the round trip returns the cropped
    # object itself: the discarded samples were field of view, not signal.
    recovered = as_numpy(mru.ifftc(cropped, axes=-1))
    assert np.allclose(recovered[0], image[0, 32:96], atol=1e-10)


def test_a_target_wider_than_the_samples_there_are_is_refused(to_array):
    with pytest.raises(ValueError, match=r"must be in \[1, 64\]"):
        mru.remove_readout_oversampling(to_array(np.ones((2, 64), dtype=complex)), 128)


def test_a_target_that_matches_is_returned_untouched(to_array):
    data = to_array(np.ones((2, 64), dtype=complex))
    assert mru.remove_readout_oversampling(data, 64) is data
