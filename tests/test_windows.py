"""Apodization windows over a centered k-space grid."""

import numpy as np
import pytest

import mrutils as mru


def test_the_fermi_window_is_flat_at_the_centre_and_gone_at_the_corner():
    window = mru.fermi_window((32, 32))
    assert window[16, 16] > 0.999
    assert window[0, 0] < 0.001


def test_the_fermi_radius_moves_where_the_roll_off_sits():
    narrow = mru.fermi_window((64, 64), radius=0.4)
    wide = mru.fermi_window((64, 64), radius=0.9)
    assert wide.sum() > narrow.sum()


def test_a_wider_transition_is_a_gentler_roll_off():
    sharp = mru.fermi_window((64, 64), radius=0.6, width=0.01)
    gentle = mru.fermi_window((64, 64), radius=0.6, width=0.2)
    edge = mru.fermi_window((64, 64), radius=0.6, width=0.2)[32, 51]
    assert 0.0 < edge < 1.0
    assert np.abs(np.gradient(gentle[32])).max() < np.abs(np.gradient(sharp[32])).max()


def test_the_hann_taper_is_one_at_the_centre_and_zero_at_its_radius():
    window = mru.hann_window((32, 32))
    assert window[16, 16] == 1.0
    assert window[0, 0] == 0.0


def test_the_hann_taper_costs_more_than_a_fermi_roll_off_of_the_same_radius():
    """It starts tapering at the origin, where Fermi is still flat."""
    assert (
        mru.hann_window((64, 64), radius=0.8).sum()
        < mru.fermi_window((64, 64), radius=0.8).sum()
    )


def test_a_window_is_ellipsoidal_on_an_anisotropic_grid():
    """Each axis reaches its own Nyquist, so no axis is clipped first."""
    window = mru.hann_window((16, 64))
    assert window[0, 32] == 0.0
    assert window[8, 0] == 0.0
    assert window[8, 32] == 1.0


def test_a_three_dimensional_window_has_three_dimensions():
    assert mru.fermi_window((8, 12, 16)).shape == (8, 12, 16)


def test_a_window_follows_the_namespace_and_device_of_like(to_array):
    reference = to_array(np.zeros((16, 16)))
    window = mru.fermi_window((16, 16), like=reference)
    assert (
        type(window).__module__.split(".")[0]
        == type(reference).__module__.split(".")[0]
    )
    if hasattr(reference, "device"):
        assert window.device == reference.device


def test_a_non_positive_transition_width_is_refused():
    with pytest.raises(ValueError, match="width must be positive"):
        mru.fermi_window((8, 8), width=0.0)


def test_a_non_positive_taper_radius_is_refused():
    with pytest.raises(ValueError, match="radius must be positive"):
        mru.hann_window((8, 8), radius=0.0)


def test_apodize_broadcasts_one_window_across_every_channel(to_array, as_numpy):
    kspace = to_array(np.ones((4, 32, 32), dtype=complex))
    apodized = as_numpy(mru.apodize(kspace, kind="hann"))
    assert apodized.shape == (4, 32, 32)
    assert np.allclose(apodized[0], apodized[3])
    assert apodized[0, 0, 0] == 0.0


def test_apodize_works_over_named_axes(to_array, as_numpy):
    kspace = to_array(np.ones((32, 4, 32), dtype=complex))
    apodized = as_numpy(mru.apodize(kspace, kind="hann", axes=(0, 2)))
    assert apodized.shape == (32, 4, 32)
    assert np.allclose(apodized[:, 0], apodized[:, 3])


def test_apodizing_suppresses_truncation_ringing():
    """The reason the window exists, measured on the point spread function.

    The PSF is evaluated on a grid four times finer than the measurement, by
    zero-padding k-space. On the measurement's own grid a hard truncation's
    side lobes fall exactly on the Dirichlet nulls and read as zero ringing,
    which is an artefact of where the samples sit, not an absence of ringing.
    """
    kspace = np.ones((64, 64), dtype=complex)
    fine = (256, 256)

    def psf(measurement):
        return np.abs(mru.ifftc(mru.resize_centered(measurement, fine)))

    def side_lobe(pattern):
        mask = np.ones(fine, dtype=bool)
        mask[120:136, 120:136] = False
        return pattern[mask].max() / pattern.max()

    sharp = side_lobe(psf(kspace))
    tapered = side_lobe(psf(mru.apodize(kspace, kind="hann")))
    assert sharp > 0.1
    assert tapered < 0.25 * sharp


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="fermi or hann"):
        mru.apodize(np.ones((8, 8), dtype=complex), kind="gaussian")
