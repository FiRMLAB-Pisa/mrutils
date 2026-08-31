"""Apodization windows over a centered k-space grid.

A reconstruction that transforms a sharply truncated k-space gets the
truncation's ringing with it. An apodization window rolls the measurement off
towards its edge instead, trading a little resolution for a point spread
function without side lobes. The two here are the ones a product
reconstruction offers: a Fermi roll-off, whose radius and width are set
separately, and a Hann taper, which is the raised cosine over the whole radius.

Both are radial in normalized k, so an anisotropic matrix gets an ellipsoid
matched to its own grid rather than a sphere that clips one axis first.
"""

from __future__ import annotations

__all__ = ["apodize", "fermi_window", "hann_window"]

from typing import Any

from ._fourier import torch_or_numpy


def _radius(shape: tuple[int, ...], like: Any | None) -> Any:
    """Return the normalized radial coordinate over a centered grid.

    Each axis runs to 1 at its own Nyquist edge, and the origin sits at index
    ``n // 2`` -- the same centre ``fftshift`` puts the DC term on, so a window
    built here multiplies centered k-space without a further shift.
    """
    xp, is_torch = torch_or_numpy(like) if like is not None else (None, False)
    if xp is None:
        from importlib import import_module

        xp = import_module("numpy")

    squared = None
    for axis, size in enumerate(shape):
        if size < 1:
            raise ValueError(f"every axis must have at least one sample, got {shape}")
        if is_torch:
            index = xp.arange(size, device=like.device, dtype=xp.float32)
        else:
            index = xp.arange(size, dtype="float64")
        half = max(size // 2, 1)
        coordinate = (index - size // 2) / half
        broadcast = [1] * len(shape)
        broadcast[axis] = size
        term = coordinate.reshape(broadcast) ** 2
        squared = term if squared is None else squared + term
    return squared**0.5


def fermi_window(
    shape: tuple[int, ...],
    *,
    radius: float = 0.9,
    width: float = 0.05,
    like: Any | None = None,
) -> Any:
    """Fermi roll-off over a centered k-space grid.

    ``1 / (1 + exp((r - radius) / width))`` on the normalized radius: flat
    inside ``radius``, falling to zero over about ``width``. Because the two
    are set separately, the passband can be kept wide while the transition
    stays gentle, which is what distinguishes it from a taper that starts
    rolling off at the origin.

    Parameters
    ----------
    shape
        The grid, two or three entries for a slice or a slab.
    radius
        Where the roll-off is half, as a fraction of the Nyquist edge.
    width
        Transition width in the same units. Smaller is sharper, and small
        enough is a truncation again.
    like
        An array whose namespace, device and dtype the window follows. NumPy
        ``float64`` when omitted.

    Returns
    -------
    array
        The window, shaped like ``shape``.

    Raises
    ------
    ValueError
        If ``width`` is not positive, or an axis has no samples.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> window = mru.fermi_window((32, 32))
    >>> window.shape
    (32, 32)

    It is flat at the centre and has fallen away by the corner:

    >>> bool(window[16, 16] > 0.99), bool(window[0, 0] < 0.01)
    (True, True)
    """
    if width <= 0.0:
        raise ValueError(f"width must be positive, got {width}")
    xp, _ = torch_or_numpy(like) if like is not None else (None, False)
    if xp is None:
        from importlib import import_module

        xp = import_module("numpy")
    return 1.0 / (1.0 + xp.exp((_radius(tuple(shape), like) - radius) / width))


def hann_window(
    shape: tuple[int, ...],
    *,
    radius: float = 1.0,
    like: Any | None = None,
) -> Any:
    """Raised-cosine taper over a centered k-space grid.

    ``0.5 * (1 + cos(pi * r / radius))`` inside ``radius`` and zero outside it.
    The taper starts at the origin, so it costs more resolution than a Fermi
    window of the same radius and leaves less ringing.

    Parameters
    ----------
    shape
        The grid, two or three entries for a slice or a slab.
    radius
        Where the taper reaches zero, as a fraction of the Nyquist edge.
    like
        An array whose namespace, device and dtype the window follows. NumPy
        ``float64`` when omitted.

    Returns
    -------
    array
        The window, shaped like ``shape``.

    Raises
    ------
    ValueError
        If ``radius`` is not positive, or an axis has no samples.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> window = mru.hann_window((32, 32))
    >>> float(window[16, 16])
    1.0

    Nothing survives beyond the radius, which is what makes it a taper rather
    than a roll-off:

    >>> bool(np.all(window[0, 0] == 0.0))
    True
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    xp, _ = torch_or_numpy(like) if like is not None else (None, False)
    if xp is None:
        from importlib import import_module

        xp = import_module("numpy")
    import math

    scaled = _radius(tuple(shape), like) / radius
    inside = scaled < 1.0
    taper = 0.5 * (1.0 + xp.cos(math.pi * scaled.clip(max=1.0)))
    return taper * inside


def apodize(
    kspace: Any,
    *,
    kind: str = "fermi",
    axes: tuple[int, ...] = (-2, -1),
    **kwargs: Any,
) -> Any:
    """Multiply centered k-space by an apodization window.

    The window is built over the axes named by ``axes`` and broadcast across
    everything else, so a ``(coils, ky, kx)`` measurement is apodized in one
    call and every channel gets the same window.

    Parameters
    ----------
    kspace
        Centered k-space, in any array namespace.
    kind
        ``"fermi"`` or ``"hann"``.
    axes
        Which axes carry the k-space grid.
    **kwargs
        Passed to the window: ``radius`` and ``width`` for Fermi, ``radius``
        for Hann.

    Returns
    -------
    array
        The apodized measurement, in the namespace of ``kspace``.

    Raises
    ------
    ValueError
        If ``kind`` names neither window.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> kspace = np.ones((4, 32, 32), dtype=complex)
    >>> apodized = mru.apodize(kspace, kind="hann")
    >>> apodized.shape
    (4, 32, 32)

    Every channel is weighted identically, and the corners are gone:

    >>> bool(np.allclose(apodized[0], apodized[3]))
    True
    >>> bool(apodized[0, 0, 0] == 0.0)
    True
    """
    builders = {"fermi": fermi_window, "hann": hann_window}
    if kind not in builders:
        raise ValueError(f"kind must be fermi or hann, got {kind!r}")

    axes = tuple(axis % kspace.ndim for axis in axes)
    shape = tuple(kspace.shape[axis] for axis in axes)
    window = builders[kind](shape, like=kspace, **kwargs)

    broadcast = [1] * kspace.ndim
    for axis, size in zip(axes, shape, strict=True):
        broadcast[axis] = size
    return kspace * window.reshape(broadcast)
