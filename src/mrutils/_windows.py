"""Apodization windows over a centered k-space grid.

A reconstruction that transforms a sharply truncated k-space gets the
truncation's ringing with it. An apodization window rolls the measurement off
towards its edge instead, trading resolution for a point spread function
without side lobes.

Both windows here follow Bernstein et al. (JMRI 2001;14:270-280): a
one-dimensional kernel ``W(u)`` in a coordinate normalized so that the window's
half-height falls at ``u = 1``, extended over a grid in one of two geometries.
The *radial* geometry evaluates the kernel on the Euclidean radius, so the
window is an ellipsoid matched to the grid's own axes; the *separable* geometry
multiplies the kernel along each axis, which keeps more of k-space's corners.
Both extend to a slab exactly as they do to a slice, and the choice between
them is a real one: radial has the higher SNR and the more isotropic point
spread function, separable the better diagonal resolution.
"""

from __future__ import annotations

__all__ = ["apodize", "fermi_window", "hann_window"]

from importlib import import_module
from typing import Any

from ._fourier import torch_or_numpy

#: Transition width of the Fermi kernel, in samples. Bernstein et al. describe
#: a product implementation using ``T = 10 / (N / 2)``, which is this many
#: samples of roll-off. The grid's *longest* axis sets it: a single scalar
#: transition cannot be ten samples of every axis at once, and taking it from
#: the shortest would put a slab's roll-off inside its own passband.
_FERMI_TRANSITION_SAMPLES = 10.0


def _namespace(like: Any | None) -> tuple[Any, bool]:
    """Return the array namespace to build in, and whether it is Torch."""
    if like is not None:
        return torch_or_numpy(like)
    return import_module("numpy"), False


def _coordinates(shape: tuple[int, ...], like: Any | None) -> list[Any]:
    """Per-axis normalized coordinates, each broadcastable over ``shape``.

    Every axis runs to 1 at its own Nyquist edge and has its origin at index
    ``n // 2`` -- the same centre ``fftshift`` puts DC on, so a window built
    here multiplies centered k-space without a further shift. Normalizing each
    axis separately is what gives an anisotropic matrix an ellipsoid matched to
    its own grid rather than a sphere that clips one axis first.
    """
    xp, is_torch = _namespace(like)
    axes = []
    for axis, size in enumerate(shape):
        if size < 1:
            raise ValueError(f"every axis must have at least one sample, got {shape}")
        if is_torch:
            index = xp.arange(size, device=like.device, dtype=xp.float32)
        else:
            index = xp.arange(size, dtype="float64")
        broadcast = [1] * len(shape)
        broadcast[axis] = size
        axes.append(((index - size // 2) / max(size // 2, 1)).reshape(broadcast))
    return axes


def _radius(shape: tuple[int, ...], like: Any | None) -> Any:
    """Euclidean radius over the normalized per-axis coordinates."""
    squared = None
    for coordinate in _coordinates(shape, like):
        term = coordinate**2
        squared = term if squared is None else squared + term
    return squared**0.5


def _extend(
    kernel: Any,
    shape: tuple[int, ...],
    like: Any | None,
    geometry: str,
) -> Any:
    """Apply a one-dimensional kernel over a grid, radially or separably."""
    if geometry == "radial":
        return kernel(_radius(shape, like))
    if geometry == "separable":
        window = None
        for coordinate in _coordinates(shape, like):
            term = kernel(abs(coordinate))
            window = term if window is None else window * term
        return window
    raise ValueError(f"geometry must be radial or separable, got {geometry!r}")


def fermi_window(
    shape: tuple[int, ...],
    *,
    radius: float = 1.0,
    width: float | None = None,
    geometry: str = "radial",
    like: Any | None = None,
) -> Any:
    """Fermi roll-off over a centered k-space grid.

    The kernel is ``1 / (1 + exp((u - radius) / width))``: flat inside
    ``radius``, falling exponentially outside it. Because the passband and the
    transition are set separately, the window can keep nearly all of the
    measurement and still roll off gently, which is what distinguishes it from
    a taper that starts at the origin.

    The default ``radius`` of 1 puts the window's half height at each axis's
    Nyquist edge, which is the normalization Bernstein et al. describe. A
    radial window is therefore exactly ``0.5`` at ``(k_max, 0)`` -- not a
    defect, but the reason a radial window leaves small side lobes along the
    axes.

    Parameters
    ----------
    shape
        The grid: two entries for a slice, three for a slab, or any number.
    radius
        Where the roll-off is half, as a fraction of the Nyquist edge.
    width
        Transition width in the same units. ``None`` is ten samples of the
        longest axis, the product setting. Smaller is sharper, and small
        enough is a truncation again. A strongly anisotropic slab gets fewer
        than ten samples of roll-off on its short axis, and one that wants
        them there should say so.
    geometry
        ``"radial"`` evaluates the kernel on the Euclidean radius, giving an
        ellipsoid; ``"separable"`` multiplies it along each axis, keeping more
        of the corners.
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
        If ``width`` is not positive, ``geometry`` names neither extension, or
        an axis has no samples.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> window = mru.fermi_window((256, 256))
    >>> window.shape
    (256, 256)

    Flat at the centre, half way at the Nyquist edge -- which is what places
    the half height there -- and gone by the corner:

    >>> float(round(window[128, 128], 3)), float(round(window[128, 0], 3))
    (1.0, 0.5)
    >>> bool(window[0, 0] < 0.01)
    True

    A slab is the same window with a third axis:

    >>> mru.fermi_window((64, 128, 128)).shape
    (64, 128, 128)
    """
    if width is None:
        width = _FERMI_TRANSITION_SAMPLES / max(max(shape) // 2, 1)
    if width <= 0.0:
        raise ValueError(f"width must be positive, got {width}")
    xp, _ = _namespace(like)

    def kernel(coordinate: Any) -> Any:
        return 1.0 / (1.0 + xp.exp((coordinate - radius) / width))

    return _extend(kernel, tuple(shape), like, geometry)


def hann_window(
    shape: tuple[int, ...],
    *,
    radius: float = 1.0,
    geometry: str = "radial",
    like: Any | None = None,
) -> Any:
    """Raised-cosine taper over a centered k-space grid.

    The kernel is ``0.5 * (1 + cos(pi * u / radius))`` inside ``radius`` and
    zero outside it. The taper starts at the origin, so it costs more
    resolution than a Fermi window of the same radius and leaves less ringing.

    Parameters
    ----------
    shape
        The grid: two entries for a slice, three for a slab, or any number.
    radius
        Where the taper reaches zero, as a fraction of the Nyquist edge.
    geometry
        ``"radial"`` or ``"separable"``, as in :func:`fermi_window`.
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
        If ``radius`` is not positive, ``geometry`` names neither extension,
        or an axis has no samples.

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
    import math

    xp, _ = _namespace(like)

    def kernel(coordinate: Any) -> Any:
        scaled = coordinate / radius
        inside = scaled < 1.0
        taper = 0.5 * (1.0 + xp.cos(math.pi * scaled.clip(max=1.0)))
        return taper * inside

    return _extend(kernel, tuple(shape), like, geometry)


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
    call and every channel gets the same window. A slab is three axes rather
    than two.

    Parameters
    ----------
    kspace
        Centered k-space, in any array namespace.
    kind
        ``"fermi"`` or ``"hann"``.
    axes
        Which axes carry the k-space grid.
    **kwargs
        Passed to the window: ``radius``, ``width`` and ``geometry`` for
        Fermi, ``radius`` and ``geometry`` for Hann.

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

    A slab names its three axes:

    >>> volume = np.ones((4, 16, 32, 32), dtype=complex)
    >>> mru.apodize(volume, axes=(-3, -2, -1)).shape
    (4, 16, 32, 32)
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
