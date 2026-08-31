"""Density compensation for a non-Cartesian trajectory."""

from __future__ import annotations

__all__ = ["pipe_menon_dcf"]

from importlib import import_module
from typing import Any


def pipe_menon_dcf(
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    backend: str = "finufft",
    **kwargs: Any,
) -> Any:
    """Estimate Pipe--Menon density-compensation weights with MRI-NUFFT.

    ``kwargs`` are passed unchanged to the selected backend's ``pipe``
    implementation, including options such as ``max_iter`` and
    normalization. The returned array remains in the array/device ecosystem
    selected by MRI-NUFFT.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> angles = np.linspace(0, np.pi, 8, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 32)
    >>> trajectory = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
    ... ).reshape(-1, 2)
    >>> weights = np.asarray(mru.pipe_menon_dcf(trajectory, (16, 16)))
    >>> weights.shape, weights.dtype.kind
    ((256,), 'f')
    """
    if len(image_shape) not in (2, 3) or any(int(item) < 1 for item in image_shape):
        raise ValueError("image_shape must contain two or three positive entries")
    try:
        density = import_module("mrinufft.density")
    except ImportError as error:
        raise ImportError(
            "Pipe-Menon DCF estimation requires mri-nufft: pip install mrutils[dcf]"
        ) from error
    weights = density.pipe(
        trajectory,
        tuple(int(item) for item in image_shape),
        backend=backend,
        **kwargs,
    )
    # The estimator works in the complex domain and answers there; a density
    # is real, and a complex one propagates into every operator that carries
    # it and into the transfer kernels built from it. The real part of a
    # complex array is a strided view, which the backends will not take, so it
    # is materialized here.
    real = getattr(weights, "real", None)
    if real is None:
        return weights
    contiguous = getattr(real, "contiguous", None)
    if callable(contiguous):
        return contiguous()
    return import_module("numpy").ascontiguousarray(real)
