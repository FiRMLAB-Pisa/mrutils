"""Centered orthonormal Fourier transforms, and centered resizing.

The one convention an MRI reconstruction means by "the Fourier transform":
``ifftshift -> fft(norm="ortho") -> fftshift``, over any set of axes, for Torch
tensors (device preserved) and NumPy arrays alike.
"""

from __future__ import annotations

__all__ = [
    "centered_fftn",
    "fftc",
    "ifftc",
    "resize_centered",
    "resize_centered_axis",
    "torch_or_numpy",
]

from importlib import import_module
from typing import Any


def torch_or_numpy(array: Any) -> tuple[Any, bool]:
    """Return the array's namespace module and whether it is Torch."""
    try:
        torch = import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(array, torch.Tensor):
        return torch, True
    return import_module("numpy"), False


def centered_fftn(data: Any, *, axes: tuple[int, ...], inverse: bool) -> Any:
    """Centered orthonormal FFT or inverse FFT over the given axes."""
    xp, is_torch = torch_or_numpy(data)
    if is_torch:
        shifted = xp.fft.ifftshift(data, dim=axes)
        transform = xp.fft.ifftn if inverse else xp.fft.fftn
        return xp.fft.fftshift(
            transform(shifted, dim=axes, norm="ortho"),
            dim=axes,
        )
    shifted = xp.fft.ifftshift(data, axes=axes)
    transform = xp.fft.ifftn if inverse else xp.fft.fftn
    return xp.fft.fftshift(
        transform(shifted, axes=axes, norm="ortho"),
        axes=axes,
    )


def resize_centered(value: Any, shape: tuple[int, ...]) -> Any:
    """Zero-pad or crop the trailing axes symmetrically about their centres."""
    xp, is_torch = torch_or_numpy(value)
    spatial_ndim = len(shape)
    result_shape = (*value.shape[:-spatial_ndim], *shape)
    result = (
        xp.zeros(result_shape, dtype=value.dtype, device=value.device)
        if is_torch
        else xp.zeros(result_shape, dtype=value.dtype)
    )
    source_slices = [slice(None)] * value.ndim
    target_slices = [slice(None)] * value.ndim
    for offset, target_size in enumerate(shape, start=value.ndim - spatial_ndim):
        source_size = value.shape[offset]
        count = min(source_size, target_size)
        source_start = (source_size - count) // 2
        target_start = (target_size - count) // 2
        source_slices[offset] = slice(source_start, source_start + count)
        target_slices[offset] = slice(target_start, target_start + count)
    result[tuple(target_slices)] = value[tuple(source_slices)]
    return result


def resize_centered_axis(value: Any, size: int, *, axis: int) -> Any:
    """Zero-pad or crop one axis symmetrically about its centre."""
    axis %= value.ndim
    if value.shape[axis] == size:
        return value
    return resize_centered(value, (size, *value.shape[axis + 1 :]))


def fftc(data: Any, *, axes: int | tuple[int, ...] = (-2, -1)) -> Any:
    """Centered orthonormal FFT over one or more axes.

    The ``ifftshift -> fft(norm="ortho") -> fftshift`` an MRI reconstruction
    means by "the Fourier transform", so a plugin states the transform once
    rather than re-deriving the shifts. Torch tensors (device preserved) and
    NumPy arrays both pass through.

    Parameters
    ----------
    data
        The array to transform.
    axes
        Axis or axes to transform over. Default is the last two.

    Returns
    -------
    array
        The transform, in the namespace of ``data``.

    See Also
    --------
    ifftc : the inverse.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> image = np.zeros((8, 8), dtype=complex)
    >>> image[4, 4] = 1.0
    >>> kspace = mru.fftc(image)

    A point at the centre of the image is flat in k-space, and the round trip
    is exact:

    >>> bool(np.allclose(np.abs(kspace), np.abs(kspace).mean()))
    True
    >>> bool(np.allclose(mru.ifftc(kspace), image, atol=1e-12))
    True
    """
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return centered_fftn(data, axes=axes, inverse=False)


def ifftc(data: Any, *, axes: int | tuple[int, ...] = (-2, -1)) -> Any:
    """Centered orthonormal inverse FFT over one or more axes.

    The inverse of :func:`fftc`; see it for the convention. A single-axis call
    along the readout decouples a Cartesian volume into independent planes.

    Parameters
    ----------
    data
        The array to transform.
    axes
        Axis or axes to transform over. Default is the last two.

    Returns
    -------
    array
        The inverse transform, in the namespace of ``data``.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> kspace = np.ones((8, 8), dtype=complex)
    >>> image = mru.ifftc(kspace)
    >>> bool(np.allclose(mru.fftc(image), kspace, atol=1e-12))
    True
    """
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return centered_fftn(data, axes=axes, inverse=True)
