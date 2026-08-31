"""Receive-array preprocessing.

Compression onto the array's principal channels, and decorrelation against a
measured noise covariance.
"""

from __future__ import annotations

__all__ = ["coil_compress", "noise_prewhiten"]

from typing import Any

from ._fourier import torch_or_numpy


def coil_compress(
    kspace: Any,
    n_coils: int | float,
    *,
    trajectory: Any | None = None,
    calibration_radius: float | None = None,
) -> tuple[Any, Any]:
    """Compress the receive array onto its principal channels.

    A receive array measures the same object through every element, so the
    channels are strongly correlated and most of what they carry lives in far
    fewer of them. The principal components of the sample covariance are those
    channels: keeping the leading ones is the linear combination that retains
    the most energy for a given count, and everything downstream -- the
    sensitivities, the solve, the buffers -- then runs on the smaller array.

    Parameters
    ----------
    kspace
        The measurement, ``(coils, samples)``. Torch tensors keep their device.
    n_coils
        Virtual channels to keep, as a count, or the fraction of the energy to
        retain when given as a float in ``(0, 1]``. A count beyond the physical
        channels keeps them all.
    trajectory
        Where each sample was taken, ``(samples, dimensions)``. Only needed
        with ``calibration_radius``.
    calibration_radius
        Estimate the components from the samples inside this fraction of the
        maximum k-space radius, rather than from every sample -- the centre is
        where the array's correlations are, and where the object is brightest.

    Returns
    -------
    compressed : array
        ``(n_coils, samples)`` in the virtual basis.
    matrix : array
        ``(n_coils, coils)``, the basis itself, for compressing anything else
        the same way -- the acquisitions that arrive after a calibration
        established it, or the sensitivities they are solved against.

    Raises
    ------
    ValueError
        If ``kspace`` is not two-dimensional, ``n_coils`` asks for nothing, or
        ``calibration_radius`` is given without a trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> rng = np.random.default_rng(0)
    >>> lines = rng.normal(size=(8, 256)) + 1j * rng.normal(size=(8, 256))

    Eight channels carrying four independent signals compress onto four
    without loss, and the basis comes back to apply to everything that
    follows:

    >>> lines[4:] = lines[:4]
    >>> compressed, basis = mru.coil_compress(lines, 4)
    >>> compressed.shape, basis.shape
    ((4, 256), (4, 8))
    """
    xp, _ = torch_or_numpy(kspace)
    if kspace.ndim != 2:
        raise ValueError(f"kspace must be (coils, samples), got {kspace.shape}")
    if (calibration_radius is None) != (trajectory is None):
        raise ValueError("calibration_radius and trajectory are given together")

    region = kspace
    if calibration_radius is not None:
        radius = xp.sqrt(xp.sum(trajectory**2, axis=-1))
        region = kspace[:, radius < calibration_radius * radius.max()]

    # Eigenvectors of the sample covariance, largest eigenvalue first. eigh
    # returns them as columns, and ascending, so the order is applied to the
    # second axis and the rows of the result are the virtual channels.
    values, vectors = xp.linalg.eigh(region @ region.conj().T)
    order = xp.argsort(-values)
    energies = [float(values[int(index)]) for index in order]

    keep = _retained_channels(n_coils, energies)
    matrix = vectors[:, order[:keep]].conj().T
    # Torch's conj() is a lazy view carrying a conjugate bit, and a tensor that
    # carries one refuses .numpy(). The basis is handed to the caller to apply
    # to arrays this function never sees, so it is materialized here.
    resolve = getattr(matrix, "resolve_conj", None)
    if resolve is not None:
        matrix = resolve()
    return matrix @ kspace, matrix


def _retained_channels(n_coils: int | float, energies: list[float]) -> int:
    """How many principal channels a count or an energy fraction asks for."""
    if isinstance(n_coils, float) and not float(n_coils).is_integer():
        if not 0.0 < n_coils <= 1.0:
            raise ValueError(f"an energy fraction must lie in (0, 1], got {n_coils}")
        total = sum(energies)
        if total <= 0.0:
            return 1
        running = 0.0
        for count, energy in enumerate(energies, start=1):
            running += energy
            if running / total >= n_coils:
                return count
        return len(energies)
    keep = int(n_coils)
    if keep < 1:
        raise ValueError(f"n_coils must keep at least one channel, got {n_coils}")
    return min(keep, len(energies))


def noise_prewhiten(
    kspace: Any,
    noise: Any,
    *,
    coil_axis: int = -2,
    scale_factor: float = 1.0,
) -> Any:
    """Decorrelate receiver coils using the measured noise covariance.

    A receive array's channels see correlated noise, and a solve that assumes
    they do not weights them wrongly. The noise scan measures that covariance;
    whitening is the Cholesky solve that turns it into the identity, so every
    channel afterwards carries unit, independent noise.

    Parameters
    ----------
    kspace
        The measurement, with the channels along ``coil_axis``.
    noise
        The noise scan, channels along the same axis.
    coil_axis
        Which axis the channels run along.
    scale_factor
        Applied to the whitened data, for a caller keeping a known noise level.

    Returns
    -------
    array
        The whitened measurement, in the namespace of ``kspace``.

    Raises
    ------
    TypeError
        If the measurement and the noise are not in the same array library.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.normal(size=(4, 512)) + 1j * rng.normal(size=(4, 512))

    Whitening the noise scan against itself leaves an identity covariance,
    which is what every readout that follows is measured against:

    >>> whitened = mru.noise_prewhiten(noise, noise, coil_axis=0)
    >>> covariance = whitened @ whitened.conj().T / whitened.shape[-1]
    >>> bool(np.allclose(covariance, np.eye(4), atol=5e-2))
    True
    """
    xp, is_torch = torch_or_numpy(kspace)
    _, noise_is_torch = torch_or_numpy(noise)
    if is_torch != noise_is_torch:
        raise TypeError("kspace and noise must use the same array library")

    if is_torch:
        moved_noise = noise.movedim(coil_axis, 0)
        noise_flat = moved_noise.reshape(moved_noise.shape[0], -1)
        covariance = noise_flat @ noise_flat.conj().transpose(0, 1)
        covariance = covariance / noise_flat.shape[-1]
        cholesky = xp.linalg.cholesky(covariance)
        moved_data = kspace.movedim(coil_axis, 0)
        flat = moved_data.reshape(moved_data.shape[0], -1)
        whitened = xp.linalg.solve_triangular(
            cholesky,
            flat,
            upper=False,
        )
        whitened = whitened * scale_factor**0.5
        return whitened.reshape(moved_data.shape).movedim(0, coil_axis)

    moved_noise = xp.moveaxis(noise, coil_axis, 0)
    noise_flat = moved_noise.reshape(moved_noise.shape[0], -1)
    covariance = noise_flat @ noise_flat.conj().T / noise_flat.shape[-1]
    cholesky = xp.linalg.cholesky(covariance)
    moved_data = xp.moveaxis(kspace, coil_axis, 0)
    flat = moved_data.reshape(moved_data.shape[0], -1)
    whitened = xp.linalg.solve(cholesky, flat) * scale_factor**0.5
    return xp.moveaxis(whitened.reshape(moved_data.shape), 0, coil_axis)
