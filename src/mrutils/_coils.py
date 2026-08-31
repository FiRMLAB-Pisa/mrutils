"""Receive-array preprocessing.

Compression onto the array's principal channels, and decorrelation against a
measured noise covariance.
"""

from __future__ import annotations

__all__ = ["apply_coil_compression", "coil_compress", "noise_prewhiten"]

from typing import Any

from ._fourier import torch_or_numpy

#: Samples per pass when none is given. The working set of a pass is this many
#: samples in every channel, so the peak above the data itself is bounded by
#: the batch rather than by the scan.
_DEFAULT_BATCH = 1 << 18


def _batches(n_samples: int, batch_size: int | None) -> range:
    """Sample offsets to work through, one pass each."""
    if batch_size is None:
        batch_size = _DEFAULT_BATCH
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return range(0, n_samples, batch_size)


def coil_compress(
    kspace: Any,
    n_coils: int | float,
    *,
    trajectory: Any | None = None,
    calibration_radius: float | None = None,
    batch_size: int | None = None,
    device: Any | None = None,
) -> tuple[Any, Any]:
    """Compress the receive array onto its principal channels.

    A receive array measures the same object through every element, so the
    channels are strongly correlated and most of what they carry lives in far
    fewer of them. The principal components of the sample covariance are those
    channels: keeping the leading ones is the linear combination that retains
    the most energy for a given count, and everything downstream -- the
    sensitivities, the solve, the buffers -- then runs on the smaller array.

    The covariance is channels by channels however many samples were taken, so
    it is accumulated a pass at a time and the measurement is never multiplied
    whole. :func:`apply_coil_compression` applies the basis the same way.

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
    batch_size
        Samples per pass, for both the covariance and the application.
        ``None`` is a fixed default; smaller holds less on ``device``.
    device
        Where to run the products, for a Torch measurement whose own device
        cannot hold the scan. As :func:`apply_coil_compression`.

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
        If ``kspace`` is not two-dimensional, ``n_coils`` asks for nothing,
        ``calibration_radius`` is given without a trajectory or selects no
        samples, or ``batch_size`` is not positive.

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

    inside = None
    if calibration_radius is not None:
        radius = xp.sqrt(xp.sum(trajectory**2, axis=-1))
        inside = radius < calibration_radius * radius.max()

    # The covariance is channels by channels however many samples there are,
    # so it is accumulated a pass at a time and the whole measurement is never
    # multiplied at once.
    covariance = None
    for start in _batches(kspace.shape[-1], batch_size):
        stop = start + (batch_size or _DEFAULT_BATCH)
        block = kspace[:, start:stop]
        if inside is not None:
            block = block[:, inside[start:stop]]
        if block.shape[-1] == 0:
            continue
        if device is not None:
            block = block.to(device)
        gram = block @ block.conj().T
        covariance = gram if covariance is None else covariance + gram
    if covariance is None:
        raise ValueError("the calibration region contains no samples")

    # Eigenvectors of the sample covariance, largest eigenvalue first. eigh
    # returns them as columns, and ascending, so the order is applied to the
    # second axis and the rows of the result are the virtual channels.
    values, vectors = xp.linalg.eigh(
        covariance.to(kspace.device) if device else covariance
    )
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
    compressed = apply_coil_compression(
        matrix, kspace, batch_size=batch_size, device=device
    )
    return compressed, matrix


def apply_coil_compression(
    matrix: Any,
    kspace: Any,
    *,
    coil_axis: int = 0,
    batch_size: int | None = None,
    device: Any | None = None,
) -> Any:
    """Apply a compression basis, optionally streaming through another device.

    A basis established on a calibration is applied to everything that
    follows. Where the scan does not fit on the accelerator, ``device`` runs
    the product there a batch at a time while the measurement and the result
    stay where they are: what the device holds is one batch and the basis, not
    the scan. Without it the product runs wherever the data already is.

    Parameters
    ----------
    matrix
        The basis, ``(virtual, coils)``, as :func:`coil_compress` returns it.
    kspace
        The measurement, channels along ``coil_axis`` and any shape besides.
    coil_axis
        Which axis the channels run along.
    batch_size
        Samples per pass. ``None`` is a fixed default; smaller holds less on
        ``device``.
    device
        Where to run the product, for Torch measurements whose own device
        cannot hold the scan. ``None`` runs it where the data is.

    Returns
    -------
    array
        The measurement in the virtual basis, shaped as it arrived except for
        the channel count, in the namespace and on the device of ``kspace``.

    Raises
    ------
    ValueError
        If the basis does not match the channels, or ``batch_size`` is not
        positive.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> rng = np.random.default_rng(0)
    >>> lines = rng.normal(size=(8, 4, 64)) + 1j * rng.normal(size=(8, 4, 64))
    >>> _, basis = mru.coil_compress(lines.reshape(8, -1), 4)
    >>> mru.apply_coil_compression(basis, lines).shape
    (4, 4, 64)
    """
    xp, is_torch = torch_or_numpy(kspace)
    moved = (
        kspace.movedim(coil_axis, 0) if is_torch else xp.moveaxis(kspace, coil_axis, 0)
    )
    if matrix.shape[-1] != moved.shape[0]:
        raise ValueError(
            f"a {tuple(matrix.shape)} basis does not fit {moved.shape[0]} channels"
        )

    flat = moved.reshape(moved.shape[0], -1)
    shape = (matrix.shape[0], *moved.shape[1:])
    dtype = xp.promote_types(matrix.dtype, flat.dtype)
    result = (
        xp.zeros((matrix.shape[0], flat.shape[-1]), dtype=dtype, device=flat.device)
        if is_torch
        else xp.zeros((matrix.shape[0], flat.shape[-1]), dtype=dtype)
    )
    span = batch_size or _DEFAULT_BATCH
    if device is not None:
        matrix = matrix.to(device)
    for start in _batches(flat.shape[-1], batch_size):
        block = flat[:, start : start + span]
        if device is None:
            result[:, start : start + span] = matrix @ block
        else:
            result[:, start : start + span] = (matrix @ block.to(device)).to(
                result.device
            )

    result = result.reshape(shape)
    return (
        result.movedim(0, coil_axis) if is_torch else xp.moveaxis(result, 0, coil_axis)
    )


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
