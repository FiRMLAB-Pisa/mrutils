"""Readout-oversampling removal.

A scanner digitises more samples along the readout than the prescribed matrix
asks for, so the anti-aliasing filter has room to roll off. Those extra samples
buy field of view rather than resolution, which is why they come off in the
image domain and not by discarding k-space.
"""

from __future__ import annotations

__all__ = ["remove_readout_oversampling"]

from typing import Any

from ._fourier import centered_fftn


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
