"""EPI ramp resampling and odd/even phase correction."""

import numpy as np
import pytest

import mrutils as mru


@pytest.fixture
def ramp_sampled():
    """A readout swept sinusoidally, and the uniform grid it belongs on."""
    samples, support = 64, 16
    sweep = np.sin(np.linspace(-np.pi / 2, np.pi / 2, samples)) / 2.0
    uniform = np.linspace(-0.5, 0.5, samples)
    return sweep, uniform, support


def _band_limited(positions, support, rng):
    """The non-uniform transform, at `positions`, of an object `support` wide."""
    grid = np.arange(support) - support // 2
    obj = rng.normal(size=support) + 1j * rng.normal(size=support)
    return np.exp(-2j * np.pi * np.outer(positions, grid)) @ obj, obj


def test_ramp_resampling_is_exact_for_a_band_limited_readout(ramp_sampled):
    sweep, uniform, support = ramp_sampled
    rng = np.random.default_rng(0)
    taken, obj = _band_limited(sweep, support, rng)
    grid = np.arange(support) - support // 2
    wanted = np.exp(-2j * np.pi * np.outer(uniform, grid)) @ obj

    operator = mru.epi_ramp_operator(sweep, uniform, support)
    resampled = operator @ taken
    error = np.linalg.norm(resampled - wanted) / np.linalg.norm(wanted)
    assert error < 1e-3


def test_ramp_resampling_beats_linear_interpolation(ramp_sampled):
    """The claim the operator exists to make, measured rather than asserted."""
    sweep, uniform, support = ramp_sampled
    rng = np.random.default_rng(1)
    taken, obj = _band_limited(sweep, support, rng)
    grid = np.arange(support) - support // 2
    wanted = np.exp(-2j * np.pi * np.outer(uniform, grid)) @ obj

    def error(estimate):
        return np.linalg.norm(estimate - wanted) / np.linalg.norm(wanted)

    resampled = mru.epi_ramp_operator(sweep, uniform, support) @ taken
    linear = np.interp(uniform, sweep, taken.real) + 1j * np.interp(
        uniform, sweep, taken.imag
    )
    assert error(resampled) < 0.1 * error(linear)


def test_the_operator_is_built_once_and_applies_to_a_whole_train(ramp_sampled):
    sweep, uniform, support = ramp_sampled
    operator = mru.epi_ramp_operator(sweep, uniform, support)
    assert operator.shape == (uniform.size, sweep.size)
    train = np.ones((8, sweep.size), dtype=np.complex64)
    assert (train @ operator.T).shape == (8, uniform.size)


def test_positions_that_do_not_describe_a_readout_are_refused():
    with pytest.raises(ValueError, match="describe a readout"):
        mru.epi_ramp_operator([0.0], [0.0, 0.1], 4)


def test_a_non_positive_support_is_refused():
    with pytest.raises(ValueError, match="support must be positive"):
        mru.epi_ramp_operator(np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 8), 0)


def _navigator_line(*, coils=4, samples=64, width=1.0):
    """A blip-nulled navigator line.

    The phase fit is weighted by hybrid-space magnitude, so what matters is
    that the line is broad *there*: a readout built as a narrow k-space feature
    leaves only a handful of samples above the weight threshold, and a
    high-order fit on those is under-determined.
    """
    profile = np.exp(-(np.linspace(-width, width, samples) ** 2)).astype(complex)
    line = mru.fftc(profile, axes=-1)
    return np.repeat(line[None], coils, axis=0).astype(np.complex64)


def _delayed(line, coefficients):
    """Put a phase polynomial on a line, in hybrid space, as a delay does."""
    hybrid = mru.ifftc(line, axes=-1)
    coordinate = np.linspace(-1.0, 1.0, hybrid.shape[-1])
    ramp = np.polynomial.polynomial.polyval(coordinate, coefficients)
    return mru.fftc(hybrid * np.exp(1j * ramp), axes=-1)


def test_the_navigator_fit_is_the_correction_not_the_impressed_phase():
    """The sign convention, stated where a caller will trip over it.

    The fit is what ``correct_lines`` adds to a reversed line, so it comes back
    as the negative of the phase the reversed line was carrying.
    """
    line = _navigator_line()
    impressed = [0.3, 0.8]
    reversed_line = _delayed(line, impressed)

    fit = mru.estimate_epi_phase([line, reversed_line, line])
    assert fit.shape == (2,)
    assert np.allclose(fit, [-value for value in impressed], atol=0.05)


def test_the_fit_undoes_the_phase_it_measured():
    line = _navigator_line()
    reversed_line = _delayed(line, [0.3, 0.8])

    fit = mru.estimate_epi_phase([line, reversed_line, line])
    corrected = mru.correct_lines([(reversed_line[..., ::-1], True)], fit)[0]
    assert np.allclose(corrected, line, atol=1e-2)


def test_a_higher_order_fit_returns_the_coefficients_it_was_asked_for():
    line = _navigator_line(coils=2)
    fit = mru.estimate_epi_phase(
        [line, _delayed(line, [0.1, 0.5]), line], polynomial_order=3
    )
    assert fit.shape == (4,)


def test_fewer_than_three_navigator_lines_is_refused():
    line = np.ones((2, 16), dtype=np.complex64)
    with pytest.raises(ValueError, match="three lines"):
        mru.estimate_epi_phase([line, line])


def test_a_negative_polynomial_order_is_refused():
    line = np.ones((2, 16), dtype=np.complex64)
    with pytest.raises(ValueError, match="non-negative"):
        mru.estimate_epi_phase([line] * 3, polynomial_order=-1)


def test_correcting_the_train_removes_the_ghost():
    """A ghost is what an uncorrected odd/even phase puts at half the field."""
    truth = np.zeros((32, 32))
    truth[10:22, 12:20] = 1.0
    kspace = mru.fftc(truth.astype(complex))
    coefficients = [0.4, 0.9]

    def backwards(line):
        return _delayed(line, coefficients)[..., ::-1]

    train = [
        (line[None], False) if index % 2 == 0 else (backwards(line[None]), True)
        for index, line in enumerate(kspace)
    ]
    middle = kspace[16][None]
    fit = mru.estimate_epi_phase([middle, backwards(middle)[..., ::-1], middle])

    def placed(phase):
        return np.stack([mru.correct_lines([line], phase)[0][0] for line in train])

    def ghost(image):
        # The ghost sits half a field of view away along the phase-encode axis.
        return np.linalg.norm(np.abs(image)[:8]) / np.linalg.norm(np.abs(image))

    flipped_only = mru.ifftc(placed(None))
    corrected = mru.ifftc(placed(fit))
    assert ghost(corrected) < 0.5 * ghost(flipped_only)


def test_a_forward_line_passes_through_untouched():
    line = (np.arange(16) + 1j).astype(np.complex64)[None]
    corrected = mru.correct_lines([(line, False)])
    assert np.allclose(corrected[0], line)


def test_a_reversed_line_is_flipped_even_without_a_phase():
    line = (np.arange(16) + 1j).astype(np.complex64)[None]
    corrected = mru.correct_lines([(line, True)])
    assert np.allclose(corrected[0], line[..., ::-1])
