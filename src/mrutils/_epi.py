"""EPI readout corrections: ramp-sampling resampling and odd/even phase.

Both are per-readout work that has nothing to do with the reconstruction that
follows, and both are learned once from a calibration and then applied to every
line of the train.
"""

from __future__ import annotations

__all__ = ["correct_lines", "epi_ramp_operator", "estimate_epi_phase"]

from typing import Any

from ._fourier import fftc, ifftc


def epi_ramp_operator(
    sample_positions: Any,
    target_positions: Any,
    support: int,
    *,
    regularization: float = 1e-6,
) -> Any:
    """Resample a readout from where it was taken onto where it belongs.

    A readout is band-limited: it is the transform of an object that occupies
    ``support`` pixels and nothing outside them. So samples taken anywhere
    determine it everywhere, and moving them onto the grid is not an
    approximation but a change of basis -- the least-squares inverse of the
    non-uniform transform, followed by the uniform one. The operator's entries
    are the sinc-like kernels that implies.

    Linear interpolation is the cheap stand-in for this and is visibly worse:
    over a readout whose ramps take half its duration, this resampling is exact
    to numerical precision where a linear one leaves seven percent.

    What makes it exact is that the samples outnumber the pixels they have to
    determine, which is what readout oversampling buys. Where they do not --
    where the fast part of the sweep steps further than ``1 / support`` -- the
    readout has aliased and no resampling recovers it; ``regularization`` keeps
    the solve from amplifying that, it does not undo it.

    One lobe is played for every readout of a train, so the operator is built
    once and applied to each.

    Parameters
    ----------
    sample_positions
        Where each sample was taken, in k, normalised so the readout spans at
        most ``[-0.5, 0.5]``. The trajectory an acquisition carries: a client
        attaches one exactly when the gradient was still moving under the ADC,
        which is when a readout needs this.
    target_positions
        Where they belong -- the uniform grid, in the same units.
    support
        Pixels the object occupies along the readout: the reconstructed matrix,
        not the oversampled one the scanner digitised.
    regularization
        Tikhonov weight on the normal equations, relative to the sample count.

    Returns
    -------
    numpy.ndarray
        ``(target, source)``. Applying it to a ``(coils, samples)`` readout is
        ``readout @ operator.T``.

    Raises
    ------
    ValueError
        If either position set does not describe a readout, or ``support`` is
        not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> sampled = np.sin(np.linspace(-np.pi / 2, np.pi / 2, 16))
    >>> uniform = np.linspace(-1, 1, 16)
    >>> operator = mru.epi_ramp_operator(sampled, uniform, 3)
    >>> operator.shape
    (16, 16)
    """
    import numpy as np

    sample_positions = np.asarray(sample_positions, dtype=float).reshape(-1)
    target_positions = np.asarray(target_positions, dtype=float).reshape(-1)
    support = int(support)
    if sample_positions.size < 2 or target_positions.size < 2:
        raise ValueError("both position sets must describe a readout")
    if support < 1:
        raise ValueError(f"support must be positive, got {support}")

    grid = np.arange(support) - support // 2
    taken = np.exp(-2j * np.pi * np.outer(sample_positions, grid))
    wanted = np.exp(-2j * np.pi * np.outer(target_positions, grid))
    normal = taken.conj().T @ taken + regularization * sample_positions.size * np.eye(
        support
    )
    return (wanted @ np.linalg.solve(normal, taken.conj().T)).astype(np.complex64)


def estimate_epi_phase(
    navigator_lines: list[Any],
    *,
    polynomial_order: int = 1,
) -> Any:
    """Fit the odd/even phase an EPI readout carries, from a blip-nulled navigator.

    Reversing a readout does not reverse the delays it was played through, so a
    line read backwards carries a phase its forward neighbours do not, and
    leaving it there is what puts a ghost at half the field of view. The
    navigator measures it directly: three blip-nulled lines of alternating
    polarity see the same object, so the phase difference between the middle
    one and its neighbours is that phase and nothing else.

    A first-order fit is the correction every product reconstruction applies,
    because the term that dominates is the gradient and ADC delay and a delay
    is linear in the readout. Raising the order picks up what eddy currents
    leave beyond it, which an oblique or a strongly driven readout has more of.

    The fit is weighted by the cross-correlation magnitude and ignores the
    samples below a tenth of its peak: a phase difference where there is no
    signal is noise, and letting the readout's empty edges into an unweighted
    fit is what drags a high-order one off.

    Parameters
    ----------
    navigator_lines : list of numpy.ndarray
        Three ``(coils, samples)`` lines, polarity ``+ - +``, reversed lines
        already flipped back into readout order.
    polynomial_order
        Order of the phase polynomial. ``1`` is the gradient-delay ramp and a
        constant.

    Returns
    -------
    numpy.ndarray
        Coefficients of the phase, lowest order first, in a coordinate running
        from ``-1`` to ``1`` across the readout -- so a fit measured on the
        navigator applies to a readout of any length.

    Raises
    ------
    ValueError
        If fewer than three navigator lines are given, or the order is
        negative.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> line = np.exp(-np.linspace(-2, 2, 32) ** 2).astype(np.complex64)
    >>> line = line[None].repeat(2, 0)
    >>> ramp = np.exp(1j * 0.15 * np.arange(32)).astype(np.complex64)
    >>> fit = mru.estimate_epi_phase([line * ramp, line, line * ramp])
    >>> fit.shape
    (2,)
    """
    import numpy as np

    if len(navigator_lines) < 3:
        raise ValueError(
            f"the navigator is three lines of alternating polarity, got "
            f"{len(navigator_lines)}"
        )
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")

    forward = 0.5 * (_hybrid(navigator_lines[0]) + _hybrid(navigator_lines[2]))
    backward = _hybrid(navigator_lines[1])
    cross = np.sum(forward * np.conj(backward), axis=0)

    weights = np.abs(cross)
    phase = np.unwrap(np.angle(cross))
    coordinate = np.linspace(-1.0, 1.0, phase.size)
    keep = weights > 0.1 * weights.max()
    return np.polynomial.polynomial.polyfit(
        coordinate[keep], phase[keep], polynomial_order, w=weights[keep]
    )


def correct_lines(lines: list[tuple[Any, bool]], phase: Any = None) -> list[Any]:
    """Flip and phase-correct a train's lines into a consistent readout.

    The forward lines define the grid, so a reversed one is rotated onto them
    rather than both being met in the middle: the correction is one-sided, and
    the image does not move.

    Parameters
    ----------
    lines : list of tuple
        ``(data, reversed)`` per line, data ``(coils, samples)``.
    phase
        The polynomial coefficients :func:`estimate_epi_phase` returned.
        ``None`` -- before a navigator has arrived -- flips a reversed line
        without demodulating it.

    Returns
    -------
    list of numpy.ndarray
        The corrected lines, all in forward readout order.

    Examples
    --------
    A reversed line carries the delay it was played through, and leaving it
    there is what puts a copy of the object at half the field of view.
    ``phase=None`` is the same train flipped but not demodulated, which is
    what the ghost is:

    >>> import numpy as np
    >>> import mrutils as mru
    >>> line = np.ones((2, 32), dtype=np.complex64)
    >>> corrected = mru.correct_lines([(line, False), (line, True)])
    >>> len(corrected), corrected[0].shape
    (2, (2, 32))
    """
    import numpy as np

    corrected = []
    for data, backwards in lines:
        row = np.asarray(data)
        if backwards:
            row = row[..., ::-1]
            if phase is not None:
                hybrid = _hybrid(row)
                coordinate = np.linspace(-1.0, 1.0, hybrid.shape[-1])
                ramp = np.polynomial.polynomial.polyval(coordinate, phase)
                row = fftc(hybrid * np.exp(1j * ramp), axes=-1)
        corrected.append(row.astype(np.complex64))
    return corrected


def _hybrid(rows: Any) -> Any:
    """Rows into hybrid space: inverse FFT along the readout."""
    import numpy as np

    return ifftc(np.asarray(rows), axes=-1)
