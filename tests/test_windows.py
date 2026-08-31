"""Apodization windows over a centered k-space grid."""

import numpy as np
import pytest

import mrutils as mru


def test_the_fermi_window_is_flat_at_the_centre_and_gone_at_the_corner():
    window = mru.fermi_window((256, 256))
    assert window[128, 128] > 0.999
    assert window[0, 0] < 0.01


def test_the_fermi_half_height_sits_on_each_axis_nyquist_edge():
    """Bernstein et al.'s normalization: the kernel's FWHM is at u = 1."""
    window = mru.fermi_window((256, 256))
    assert window[128, 0] == pytest.approx(0.5)
    assert window[0, 128] == pytest.approx(0.5)


def test_the_default_transition_is_ten_samples_of_the_longest_axis():
    """T = 10 / (N / 2), the product setting the paper describes."""
    for shape in ((256, 256), (64, 256), (32, 128, 128)):
        stated = mru.fermi_window(shape, width=10.0 / (max(shape) // 2))
        assert np.allclose(mru.fermi_window(shape), stated)


def test_the_default_transition_leaves_a_slab_flat_at_dc():
    """Taken from the shortest axis instead, the roll-off eats the passband."""
    window = mru.fermi_window((32, 256, 256))
    assert window[16, 128, 128] > 0.999


def test_the_radial_geometry_apodizes_the_corners_as_the_paper_measures():
    """Reproduces Eq. 13: the diagonal Nyquist point, radial over separable.

    Evaluating the window at DC with the radius moved to ``1 - u`` is the
    one-dimensional kernel at ``u``, which is what both geometries are built
    from. The paper's N = 256 transition width gives 52.4% in two dimensions
    and 50.7% in three.
    """
    transition = 10.0 / 128

    def kernel(u):
        return float(mru.fermi_window((4, 4), radius=1.0 - u, width=transition)[2, 2])

    # The radial window is exactly one half at the diagonal point, since the
    # point sits at u = 1 whichever dimension it is reached in.
    assert 0.5 / kernel(1 / np.sqrt(2)) ** 2 == pytest.approx(0.524, abs=5e-4)
    assert 0.5 / kernel(1 / np.sqrt(3)) ** 3 == pytest.approx(0.507, abs=5e-4)


def test_the_separable_geometry_keeps_more_of_the_corners_than_the_radial_one():
    """The paper's claim is about the corners, and it sharpens with dimension.

    It is not a claim about the whole window: each axis contributes a factor
    below one, so on a grid small enough for the default transition to be a
    large fraction of it, the separable geometry attenuates the passband more
    than the radial one does.
    """
    ratios = []
    for shape in ((128, 128), (128, 128, 128)):
        corner = (0,) * len(shape)
        radial = mru.fermi_window(shape, geometry="radial")[corner]
        separable = mru.fermi_window(shape, geometry="separable")[corner]
        assert separable > radial
        ratios.append(radial / separable)
    assert ratios[1] < ratios[0]


def test_a_geometry_that_names_neither_extension_is_refused():
    with pytest.raises(ValueError, match="radial or separable"):
        mru.fermi_window((8, 8), geometry="cartesian")


def test_a_three_dimensional_window_is_an_ellipsoid_on_its_own_axes():
    """Each axis reaches its own Nyquist, in a slab as in a slice."""
    window = mru.hann_window((8, 16, 64))
    assert window[4, 8, 32] == 1.0
    for corner in ((0, 8, 32), (4, 0, 32), (4, 8, 0)):
        assert window[corner] == 0.0


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
