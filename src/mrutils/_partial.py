"""Partial-Fourier reconstruction, and readout-oversampling removal.

Both estimators here rest on the same fact: an image whose phase varies slowly
is nearly conjugate symmetric in k-space, so an omitted k-space edge is implied
by the acquired one.
"""

from __future__ import annotations

__all__ = [
    "POCS",
    "Homodyne",
    "fill_partial_echo",
    "remove_readout_oversampling",
]

from typing import Any

from ._fourier import centered_fftn, torch_or_numpy


class Homodyne:
    """Homodyne reconstruction for one-sided Cartesian partial Fourier.

    Parameters
    ----------
    dimension
        Number of spatial Fourier dimensions.
    partial_axis
        Axis containing the partial-Fourier acquisition.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> truncated = np.zeros((1, 16, 16), dtype=complex)
    >>> truncated[..., :10] = 1.0
    >>> readout = np.zeros(16)
    >>> readout[:10] = 1.0
    >>> mru.Homodyne(dimension=2, partial_axis=-1)(truncated, readout).shape
    (1, 16, 16)

    One pass rather than POCS's iteration; ``fill_partial_echo`` reaches it by
    name, and its figure is the two side by side.
    """

    def __init__(self, *, dimension: int = 2, partial_axis: int = -2) -> None:
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")
        self.dimension = int(dimension)
        self.partial_axis = int(partial_axis)

    def __call__(self, kspace: Any, mask: Any | None = None) -> Any:
        """Reconstruct an image from partial-Fourier Cartesian k-space."""
        axes, partial_axis = _spatial_axes(
            kspace.ndim,
            self.dimension,
            self.partial_axis,
        )
        acquired = _partial_fourier_mask(kspace, mask, partial_axis)
        lowpass, weight = _homodyne_masks(acquired)
        lowpass = _broadcast_line(lowpass, kspace.ndim, partial_axis)
        weight = _broadcast_line(weight, kspace.ndim, partial_axis)
        reference = centered_fftn(kspace * lowpass, axes=axes, inverse=True)
        phase = _unit_phase(reference)
        weighted = centered_fftn(kspace * weight, axes=axes, inverse=True)
        projected = (weighted * phase.conj()).real
        return projected * phase


class POCS:
    """Projection-onto-convex-sets partial-Fourier reconstruction.

    Parameters
    ----------
    dimension
        Number of spatial Fourier dimensions.
    partial_axis
        Axis containing the partial-Fourier acquisition.
    iterations
        Maximum number of data-consistency/phase-projection iterations.
    tolerance
        Relative iterate-change tolerance. Set to zero for a fixed count.
    positive
        Also project the demodulated image onto the non-negative real cone.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> truncated = np.zeros((1, 16, 16), dtype=complex)
    >>> truncated[..., :10] = 1.0
    >>> readout = np.zeros(16)
    >>> readout[:10] = 1.0
    >>> mru.POCS(dimension=2, partial_axis=-1, iterations=4)(truncated, readout).shape
    (1, 16, 16)

    ``fill_partial_echo`` reaches this by name, and its figure is the two
    estimators side by side.
    """

    def __init__(
        self,
        *,
        dimension: int = 2,
        partial_axis: int = -2,
        iterations: int = 12,
        tolerance: float = 1e-5,
        positive: bool = True,
    ) -> None:
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.dimension = int(dimension)
        self.partial_axis = int(partial_axis)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.positive = bool(positive)

    def __call__(self, kspace: Any, mask: Any | None = None) -> Any:
        """Reconstruct an image while preserving every acquired sample."""
        xp, is_torch = torch_or_numpy(kspace)
        axes, partial_axis = _spatial_axes(
            kspace.ndim,
            self.dimension,
            self.partial_axis,
        )
        acquired = _partial_fourier_mask(kspace, mask, partial_axis)
        lowpass, _ = _homodyne_masks(acquired)
        acquired = _broadcast_line(acquired, kspace.ndim, partial_axis)
        lowpass = _broadcast_line(lowpass, kspace.ndim, partial_axis)
        phase = _unit_phase(centered_fftn(kspace * lowpass, axes=axes, inverse=True))
        estimate = kspace.copy() if not is_torch else kspace.clone()
        previous = centered_fftn(estimate, axes=axes, inverse=True)
        for _ in range(self.iterations):
            demodulated = (previous * phase.conj()).real
            if self.positive:
                demodulated = (
                    demodulated.clamp_min(0) if is_torch else xp.maximum(demodulated, 0)
                )
            projected = demodulated * phase
            proposed = centered_fftn(projected, axes=axes, inverse=False)
            estimate = xp.where(acquired, kspace, proposed)
            updated = centered_fftn(estimate, axes=axes, inverse=True)
            if self.tolerance and _relative_change(updated, previous) <= self.tolerance:
                previous = updated
                break
            previous = updated
        return previous


def fill_partial_echo(
    kspace: Any,
    readout: Any,
    iterations: int = 12,
    *,
    dimension: int,
    method: str = "pocs",
) -> Any:
    """Recover the readout edge a partial echo never acquired.

    Both estimators rest on the same fact -- an image whose phase varies slowly
    is nearly conjugate symmetric in k-space, so the missing edge is implied by
    the acquired one. :class:`POCS` iterates towards an image that reproduces
    every acquired sample; :class:`Homodyne` reaches an answer in one pass by
    weighting the acquired half and demodulating the low-resolution phase.
    POCS is the more faithful of the two and Homodyne the cheaper, which is the
    choice a scanner-side reconstruction is actually making.

    Parameters
    ----------
    kspace
        K-space over the full readout width, coil-wise or combined.
    readout
        Which readout samples were acquired, over the full width.
    iterations
        POCS iterations. Homodyne takes one pass and ignores this.
    dimension
        How many trailing axes of ``kspace`` are spatial: 2 for a slice, 3 for
        a slab. Required rather than inferred, because whether a leading axis
        is coils or partitions is the caller's to know, and guessing it wrong
        fills the wrong axis and says nothing.
    method
        ``"pocs"`` or ``"homodyne"``.

    Returns
    -------
    array
        The partial-Fourier image, in the namespace of ``kspace``: for POCS,
        the reconstruction whose re-encoding reproduces every acquired sample.

    Raises
    ------
    ValueError
        If ``method`` names neither estimator.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> image = np.zeros((16, 16), dtype=complex)
    >>> image[6:10, 6:10] = 1.0
    >>> truncated = mru.fftc(image)
    >>> truncated[:, 12:] = 0
    >>> readout = np.ones(16)
    >>> readout[12:] = 0

    Both estimators answer for the same truncation, and the name is what a
    plugin exposes as its ``partial_fourier`` setting:

    >>> pocs = mru.fill_partial_echo(truncated[None], readout, dimension=2)
    >>> homodyne = mru.fill_partial_echo(
    ...     truncated[None], readout, dimension=2, method="homodyne"
    ... )
    >>> pocs.shape == homodyne.shape
    True

    What the truncation costs, and what the conjugate symmetry buys back:

    """
    if method == "pocs":
        return POCS(dimension=dimension, partial_axis=-1, iterations=iterations)(
            kspace, readout
        )
    if method == "homodyne":
        return Homodyne(dimension=dimension, partial_axis=-1)(kspace, readout)
    raise ValueError(f"method must be pocs or homodyne, got {method!r}")


def remove_readout_oversampling(
    data: Any,
    target_size: int,
    *,
    readout_axis: int = -1,
) -> Any:
    """Remove readout oversampling by centered image-domain cropping.

    A scanner digitises more samples than the prescribed matrix so the readout
    filter has room to roll off, and those extra samples are field of view, not
    resolution: the crop is in the image domain, and what comes back is the
    same k-space over the prescribed width.

    Parameters
    ----------
    data
        K-space with the readout along ``readout_axis``.
    target_size
        Samples the prescribed matrix asks for --
        the reconstructed matrix's last entry.
    readout_axis
        Which axis the readout runs along.

    Returns
    -------
    array
        K-space over ``target_size`` samples, in the namespace of ``data``.

    Raises
    ------
    ValueError
        If ``target_size`` is not within the samples there are.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> readout = mru.ifftc(np.ones((4, 128)), axes=-1)
    >>> mru.remove_readout_oversampling(readout, 64).shape
    (4, 64)
    """
    current = data.shape[readout_axis]
    if not 0 < target_size <= current:
        raise ValueError(f"target_size must be in [1, {current}], got {target_size}")
    if target_size == current:
        return data
    image = centered_fftn(data, axes=(readout_axis,), inverse=True)
    start = (current - target_size) // 2
    selection = [slice(None)] * image.ndim
    selection[readout_axis] = slice(start, start + target_size)
    cropped = image[tuple(selection)]
    return centered_fftn(cropped, axes=(readout_axis,), inverse=False)


def _spatial_axes(
    ndim: int,
    dimension: int,
    partial_axis: int,
) -> tuple[tuple[int, ...], int]:
    if ndim < dimension:
        raise ValueError("input has fewer dimensions than the spatial transform")
    axes = tuple(range(ndim - dimension, ndim))
    selected = partial_axis % ndim
    if selected not in axes:
        raise ValueError("partial_axis must be one of the spatial dimensions")
    return axes, selected


def _partial_fourier_mask(data: Any, mask: Any | None, axis: int) -> Any:
    xp, is_torch = torch_or_numpy(data)
    if mask is None:
        occupied = data.abs() > 0 if is_torch else xp.abs(data) > 0
        reduce_axes = tuple(index for index in range(data.ndim) if index != axis)
        acquired = (
            occupied.any(dim=reduce_axes)
            if is_torch
            else occupied.any(axis=reduce_axes)
        )
    else:
        acquired = (
            xp.as_tensor(mask, device=data.device, dtype=xp.bool)
            if is_torch
            else xp.asarray(mask, dtype=bool)
        )
    if acquired.ndim != 1 or acquired.shape[0] != data.shape[axis]:
        raise ValueError("partial-Fourier mask must be one-dimensional")
    indices = (
        acquired.nonzero(as_tuple=False).reshape(-1)
        if is_torch
        else xp.flatnonzero(acquired)
    )
    count = indices.numel() if is_torch else indices.size
    if count == 0:
        raise ValueError("partial-Fourier mask contains no acquired samples")
    first = int(indices[0])
    last = int(indices[-1])
    if count != last - first + 1:
        raise ValueError("partial-Fourier samples must form one contiguous interval")
    if first != 0 and last != acquired.shape[0] - 1:
        raise ValueError("partial-Fourier samples must omit only one k-space edge")
    center = acquired.shape[0] // 2
    if not bool(acquired[center]):
        raise ValueError("partial-Fourier samples must include the k-space center")
    return acquired


def _homodyne_masks(acquired: Any) -> tuple[Any, Any]:
    xp, is_torch = torch_or_numpy(acquired)
    count = acquired.shape[0]
    indices = xp.arange(count, device=acquired.device) if is_torch else xp.arange(count)
    center = count // 2
    partner = (2 * center - indices) % count
    symmetric = acquired & acquired[partner]
    weight = acquired.to(dtype=xp.float32) if is_torch else acquired.astype(float)
    partner_values = (
        acquired[partner].to(dtype=weight.dtype)
        if is_torch
        else acquired[partner].astype(weight.dtype)
    )
    weight = weight * (2 - partner_values)
    return symmetric, weight


def _broadcast_line(line: Any, ndim: int, axis: int) -> Any:
    shape = [1] * ndim
    shape[axis] = line.shape[0]
    return line.reshape(shape)


def _unit_phase(image: Any) -> Any:
    xp, is_torch = torch_or_numpy(image)
    magnitude = image.abs() if is_torch else xp.abs(image)
    epsilon = xp.finfo(image.real.dtype).eps
    return (
        image / magnitude.clip(min=epsilon)
        if is_torch
        else image
        / xp.clip(
            magnitude,
            epsilon,
            None,
        )
    )


def _relative_change(current: Any, previous: Any) -> float:
    xp, is_torch = torch_or_numpy(current)
    if is_torch:
        numerator = xp.linalg.vector_norm((current - previous).reshape(-1))
        denominator = xp.linalg.vector_norm(previous.reshape(-1)).clamp_min(1e-12)
        return float((numerator / denominator).detach().cpu())
    numerator = xp.linalg.norm((current - previous).reshape(-1))
    denominator = max(float(xp.linalg.norm(previous.reshape(-1))), 1e-12)
    return float(numerator / denominator)
