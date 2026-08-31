"""Off-resonance measured from a multi-echo acquisition."""

from __future__ import annotations

__all__ = ["field_map"]

from typing import Any

from ._fourier import torch_or_numpy


def field_map(images: Any, echo_times: Any, *, smoothing: float = 2.0) -> Any:
    """Off-resonance in hertz, from images of one object at several echo times.

    Between two echoes a voxel turns by its own offset from the transmit
    frequency, so the phase the pair differ by, divided by the time between
    them, is that offset. Echo pairs are summed before the angle is taken:
    that is a coherent average, which a weighted mean of separately-wrapped
    angles is not.

    What the map can state is bounded by the spacing: an offset beyond
    ``+-1 / (2 * dTE)`` Hz turns by more than half a cycle between echoes and
    comes back as its alias. Spacing the echoes further apart narrows that
    range and refines what falls inside it.

    Parameters
    ----------
    images
        Complex images of one object, echo first: ``(echoes, *grid)``. NumPy
        or Torch; the result follows.
    echo_times
        When each was read, in seconds. At least two, evenly spaced -- an
        uneven train is several different ``dTE`` and so several different
        aliases, which one map cannot carry.
    smoothing
        Width in voxels of the Gaussian the echo-pair product is blurred by
        before its angle is taken. Smoothing the product rather than the map
        is what keeps a wrap from being averaged across. Zero smooths nothing.

    Returns
    -------
    array
        Off-resonance in Hz over the grid, shaped like one image.

    Raises
    ------
    ValueError
        If fewer than two echoes are given, if the number of echo times does
        not match, or if the train is not evenly spaced.

    Examples
    --------
    A field that is off by 40 Hz on one side, measured from two echoes 2 ms
    apart:

    >>> import numpy as np
    >>> import mrutils as mru
    >>> offset = np.zeros((8, 8), dtype=np.float32)
    >>> offset[:, 4:] = 40.0
    >>> echo_times = [5e-3, 7e-3]
    >>> images = np.stack(
    ...     [np.exp(2j * np.pi * offset * te) for te in echo_times]
    ... )
    >>> measured = mru.field_map(images, echo_times, smoothing=0.0)
    >>> bool(np.allclose(measured, offset, atol=1e-3))
    True
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    times = np.asarray(echo_times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        raise ValueError("a field map needs at least two echoes")
    spacing = np.diff(times)
    if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-12):
        raise ValueError(
            "echo_times must be evenly spaced: each spacing is its own "
            f"aliasing range, and these differ ({spacing.tolist()})"
        )
    xp, is_torch = torch_or_numpy(images)
    if int(images.shape[0]) != times.size:
        raise ValueError(
            f"{images.shape[0]} echoes and {times.size} echo times do not match"
        )

    product = xp.sum(images[1:] * xp.conj(images[:-1]), axis=0)
    if smoothing > 0.0:
        host = product.detach().cpu().numpy() if is_torch else np.asarray(product)
        blurred = gaussian_filter(host.real, smoothing) + 1j * gaussian_filter(
            host.imag, smoothing
        )
        product = (
            xp.as_tensor(blurred, device=product.device).to(product.dtype)
            if is_torch
            else blurred.astype(host.dtype)
        )
    angle = xp.angle(product)
    return angle / (2.0 * np.pi * float(spacing[0]))
