"""Partial-Fourier filling, and readout-oversampling removal."""

import numpy as np
import pytest

import mrutils as mru


@pytest.fixture
def truncated():
    """A phantom-ish object, its k-space, and which readout samples survived."""
    image = np.zeros((32, 32), dtype=complex)
    image[10:22, 12:20] = 1.0
    image[14:18, 8:24] = 0.6
    # A slowly varying phase is what conjugate symmetry needs; a constant one
    # would make the test pass for a reason the method does not rely on.
    ramp = np.linspace(-0.4, 0.4, 32)
    image = image * np.exp(1j * (ramp[:, None] + ramp[None, :]))
    kspace = mru.fftc(image)
    readout = np.ones(32)
    readout[24:] = 0.0
    kspace = kspace * readout[None, :]
    return image, kspace[None], readout


def test_pocs_reproduces_every_acquired_sample(truncated):
    _, kspace, readout = truncated
    filled = mru.fill_partial_echo(kspace, readout, dimension=2)
    reencoded = mru.fftc(filled)
    acquired = readout.astype(bool)
    assert np.allclose(reencoded[..., acquired], kspace[..., acquired], atol=1e-5)


def test_both_estimators_beat_zero_filling(truncated):
    image, kspace, readout = truncated

    def error(estimate):
        return np.linalg.norm(np.abs(estimate) - np.abs(image)) / np.linalg.norm(image)

    zero_filled = error(mru.ifftc(kspace)[0])
    pocs = error(mru.fill_partial_echo(kspace, readout, dimension=2)[0])
    homodyne = error(
        mru.fill_partial_echo(kspace, readout, dimension=2, method="homodyne")[0]
    )
    assert pocs < zero_filled
    assert homodyne < zero_filled


def test_the_two_estimators_answer_on_the_same_grid(truncated):
    _, kspace, readout = truncated
    pocs = mru.fill_partial_echo(kspace, readout, dimension=2)
    homodyne = mru.fill_partial_echo(kspace, readout, dimension=2, method="homodyne")
    assert pocs.shape == homodyne.shape == kspace.shape


def test_a_mask_that_omits_the_kspace_centre_is_refused(truncated):
    _, kspace, _ = truncated
    edge_only = np.zeros(32)
    edge_only[:8] = 1.0
    with pytest.raises(ValueError, match="k-space center"):
        mru.fill_partial_echo(kspace, edge_only, dimension=2)


def test_a_mask_with_a_gap_is_refused(truncated):
    _, kspace, _ = truncated
    gapped = np.ones(32)
    gapped[10:12] = 0.0
    with pytest.raises(ValueError, match="contiguous"):
        mru.fill_partial_echo(kspace, gapped, dimension=2)


def test_omitting_both_edges_is_refused(truncated):
    _, kspace, _ = truncated
    both = np.zeros(32)
    both[8:24] = 1.0
    with pytest.raises(ValueError, match="only one k-space edge"):
        mru.fill_partial_echo(kspace, both, dimension=2)


def test_an_unknown_method_is_refused(truncated):
    _, kspace, readout = truncated
    with pytest.raises(ValueError, match="pocs or homodyne"):
        mru.fill_partial_echo(kspace, readout, dimension=2, method="nope")


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
