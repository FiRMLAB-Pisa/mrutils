"""EPI readout corrections: ramp-sampling resampling and odd/even phase.

Both are per-readout work that has nothing to do with the reconstruction that
follows, and both are learned once from a calibration and then applied to every
line of the train.
"""

from __future__ import annotations

__all__ = ["correct_lines", "epi_ramp_operator", "estimate_epi_phase"]

from importlib import import_module
from typing import Any

from ._fourier import fftc, ifftc, torch_or_numpy


def _like(reference: Any) -> tuple[Any, bool]:
    """Return the namespace of ``reference``, defaulting to NumPy."""
    if reference is None:
        return import_module("numpy"), False
    return torch_or_numpy(reference)


def _as_array(value: Any) -> Any:
    """Leave an array where it is; bring anything else into NumPy."""
    if hasattr(value, "reshape") and hasattr(value, "shape"):
        return value
    return import_module("numpy").asarray(value)


def _complex(array: Any, is_torch: bool) -> Any:
    """Cast to single-precision complex in either namespace."""
    return array.to(_TORCH_COMPLEX()) if is_torch else array.astype("complex64")


def _TORCH_COMPLEX() -> Any:
    return import_module("torch").complex64


def _ramp(size: int, like: Any) -> Any:
    """Return a coordinate from -1 to 1 across ``size`` samples, beside ``like``."""
    xp, is_torch = _like(like)
    if is_torch:
        return xp.linspace(-1.0, 1.0, size, device=like.device, dtype=xp.float32)
    return xp.linspace(-1.0, 1.0, size)


def _unwrap(phase: Any) -> Any:
    """Remove the 2*pi jumps from a phase, as ``numpy.unwrap`` does."""
    import math

    step = phase[1:] - phase[:-1]
    wrapped = (step + math.pi) % (2 * math.pi) - math.pi
    turned = (wrapped == -math.pi) & (step > 0)
    wrapped = wrapped * (~turned) + math.pi * turned
    correction = (wrapped - step) * (abs(step) >= math.pi)
    total = phase * 0.0
    total[1:] = correction.cumsum(0)
    return phase + total


def _polyfit(coordinate: Any, values: Any, weights: Any, order: int) -> Any:
    """Weighted least-squares polynomial fit, lowest order first.

    The weights multiply both sides of the system, which is what
    ``numpy.polynomial.polynomial.polyfit`` does with its ``w``.
    """
    xp, is_torch = _like(coordinate)
    columns = [coordinate**power for power in range(order + 1)]
    design = xp.stack(columns, dim=-1) if is_torch else xp.stack(columns, axis=-1)
    scaled = design * weights[:, None]
    normal = scaled.T @ scaled
    right = scaled.T @ (values * weights)
    return xp.linalg.solve(normal, right)


def _polyval(coordinate: Any, coefficients: Any) -> Any:
    """Evaluate a polynomial whose coefficients run lowest order first."""
    total = coordinate * 0.0
    for power in range(len(coefficients) - 1, -1, -1):
        total = total * coordinate + coefficients[power]
    return total


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
    array
        ``(target, source)``, in the namespace and on the device of
        ``sample_positions``. Applying it to a ``(coils, samples)`` readout is
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
    sample_positions = _as_array(sample_positions)
    target_positions = _as_array(target_positions)
    xp, is_torch = _like(sample_positions)
    sample_positions = sample_positions.reshape(-1)
    target_positions = target_positions.reshape(-1)
    support = int(support)
    if sample_positions.shape[0] < 2 or target_positions.shape[0] < 2:
        raise ValueError("both position sets must describe a readout")
    if support < 1:
        raise ValueError(f"support must be positive, got {support}")

    import math

    if is_torch:
        grid = xp.arange(support, device=sample_positions.device) - support // 2
        grid = grid.to(sample_positions.dtype)
        identity = xp.eye(support, device=sample_positions.device)
    else:
        grid = xp.arange(support) - support // 2
        identity = xp.eye(support)

    taken = xp.exp(-2j * math.pi * xp.outer(sample_positions, grid))
    wanted = xp.exp(-2j * math.pi * xp.outer(target_positions, grid))
    weight = regularization * sample_positions.shape[0]
    normal = taken.conj().T @ taken + weight * _complex(identity, is_torch)
    return _complex(wanted @ xp.linalg.solve(normal, taken.conj().T), is_torch)


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
    navigator_lines : list of array
        Three ``(coils, samples)`` lines, polarity ``+ - +``, reversed lines
        already flipped back into readout order. Torch tensors keep their
        device, and the fit comes back beside them.
    polynomial_order
        Order of the phase polynomial. ``1`` is the gradient-delay ramp and a
        constant.

    Returns
    -------
    array
        Coefficients of the phase, lowest order first, in a coordinate running
        from ``-1`` to ``1`` across the readout -- so a fit measured on the
        navigator applies to a readout of any length. The constant term is
        fixed only modulo ``2 * pi``: which branch the unwrap lands on follows
        the last bits of the transform, so two fits of the same navigator can
        differ by a turn and still be the same rotation.

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
    if len(navigator_lines) < 3:
        raise ValueError(
            f"the navigator is three lines of alternating polarity, got "
            f"{len(navigator_lines)}"
        )
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")

    xp, _ = _like(_as_array(navigator_lines[0]))
    forward = 0.5 * (_hybrid(navigator_lines[0]) + _hybrid(navigator_lines[2]))
    backward = _hybrid(navigator_lines[1])
    cross = (forward * backward.conj()).sum(0)

    weights = abs(cross)
    phase = _unwrap(xp.angle(cross))
    coordinate = _ramp(phase.shape[0], phase)
    keep = weights > 0.1 * weights.max()
    return _polyfit(coordinate[keep], phase[keep], weights[keep], polynomial_order)


def correct_lines(lines: list[tuple[Any, bool]], phase: Any = None) -> list[Any]:
    """Flip and phase-correct a train's lines into a consistent readout.

    The forward lines define the grid, so a reversed one is rotated onto them
    rather than both being met in the middle: the correction is one-sided, and
    the image does not move.

    Parameters
    ----------
    lines : list of tuple
        ``(data, reversed)`` per line, data ``(coils, samples)``. Torch
        tensors keep their device.
    phase
        The polynomial coefficients :func:`estimate_epi_phase` returned.
        ``None`` -- before a navigator has arrived -- flips a reversed line
        without demodulating it.

    Returns
    -------
    list of array
        The corrected lines, all in forward readout order, each beside the
        line it came from.

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
    corrected = []
    for data, backwards in lines:
        row = _as_array(data)
        xp, is_torch = _like(row)
        if backwards:
            row = xp.flip(row, [-1]) if is_torch else row[..., ::-1]
            if phase is not None:
                hybrid = _hybrid(row)
                coordinate = _ramp(hybrid.shape[-1], hybrid)
                ramp = _polyval(coordinate, phase)
                row = fftc(hybrid * xp.exp(1j * ramp), axes=-1)
        corrected.append(_complex(row, is_torch))
    return corrected


def _hybrid(rows: Any) -> Any:
    """Rows into hybrid space: inverse FFT along the readout."""
    return ifftc(_as_array(rows), axes=-1)
