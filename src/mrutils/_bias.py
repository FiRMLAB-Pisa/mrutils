"""Receive-field shading correction.

A surface array sees the object through a sensitivity that falls off with
distance, so a uniform object comes back bright near the coils and dim in the
middle. The shading is smooth and multiplicative, which is what lets it be
separated from anatomy at all: N4 estimates it as a B-spline field by
sharpening the log-intensity histogram, and dividing it out leaves the
anatomy.

This is an image-domain correction applied after coil combination. It is not a
substitute for sensitivity maps in a SENSE solve, which is where the same
physics is handled quantitatively.
"""

from __future__ import annotations

__all__ = ["bias_field_correct"]

from importlib import import_module
from typing import Any


def _simpleitk() -> Any:
    """Import SimpleITK, or say which extra provides it."""
    try:
        return import_module("SimpleITK")
    except ImportError as error:
        raise ImportError(
            "bias field correction requires SimpleITK: pip install mrutils[bias]"
        ) from error


def bias_field_correct(
    image: Any,
    *,
    mask: Any | None = None,
    shrink_factor: int = 4,
    iterations: tuple[int, ...] = (50, 50, 50, 50),
    fitting_levels: int | None = None,
    return_field: bool = False,
) -> Any:
    """Divide out the smooth multiplicative shading of a magnitude image.

    The field is estimated on a grid coarsened by ``shrink_factor`` and then
    evaluated back at full resolution: the field is smooth by construction, so
    estimating it at full resolution costs time and buys nothing.

    Parameters
    ----------
    image
        Magnitude image or volume, two- or three-dimensional. NumPy or Torch;
        the result follows, on the same device.
    mask
        Where to estimate the field, same shape as ``image``. ``None`` uses
        Otsu's threshold on the image itself, which is what excludes air --
        estimating over background is what drags a field towards noise.
    shrink_factor
        How much to coarsen the grid the field is fitted on.
    iterations
        Maximum iterations at each fitting level, coarsest first.
    fitting_levels
        Number of multi-resolution levels. ``None`` takes it from the length
        of ``iterations``.
    return_field
        Also return the estimated field.

    Returns
    -------
    corrected : array
        The image divided by the field, in the namespace of ``image``.
    field : array
        The field itself, returned only when ``return_field`` is set.

    Raises
    ------
    ImportError
        If SimpleITK is not installed.
    ValueError
        If ``image`` is not two- or three-dimensional, or ``shrink_factor`` is
        not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import mrutils as mru
    >>> rng = np.random.default_rng(0)
    >>> truth = np.zeros((32, 32), dtype=np.float32)
    >>> truth[8:24, 8:24] = 1.0

    A left-to-right shading over a uniform block, and what is left after it is
    divided out:

    >>> ramp = np.linspace(0.5, 1.5, 32, dtype=np.float32)[None]
    >>> observed = truth * ramp
    >>> corrected = mru.bias_field_correct(observed)
    >>> inside = truth > 0
    >>> bool(corrected[inside].std() < observed[inside].std())
    True
    """
    import numpy as np

    sitk = _simpleitk()

    if shrink_factor < 1:
        raise ValueError(f"shrink_factor must be positive, got {shrink_factor}")

    is_torch = type(image).__module__.startswith("torch")
    host = image.detach().cpu().numpy() if is_torch else np.asarray(image)
    if host.ndim not in {2, 3}:
        raise ValueError(f"image must be 2D or 3D, got shape {host.shape}")

    volume = sitk.GetImageFromArray(host.astype(np.float32))
    if mask is None:
        mask_volume = sitk.OtsuThreshold(volume, 0, 1, 200)
    else:
        host_mask = mask.detach().cpu().numpy() if is_torch else np.asarray(mask)
        mask_volume = sitk.GetImageFromArray(host_mask.astype(np.uint8))

    levels = len(iterations) if fitting_levels is None else int(fitting_levels)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(
        [int(count) for count in iterations][:levels]
    )

    if shrink_factor > 1:
        shrunk = sitk.Shrink(volume, [shrink_factor] * volume.GetDimension())
        shrunk_mask = sitk.Shrink(mask_volume, [shrink_factor] * volume.GetDimension())
        corrector.Execute(shrunk, shrunk_mask)
    else:
        corrector.Execute(volume, mask_volume)

    # The field is fitted on the coarse grid but evaluated on the full one, so
    # the correction is applied at the resolution the image actually has.
    log_field = corrector.GetLogBiasFieldAsImage(volume)
    field = sitk.GetArrayFromImage(sitk.Exp(log_field)).astype(host.dtype)
    corrected = np.divide(host, field, out=np.zeros_like(host), where=field != 0)

    if is_torch:
        import torch

        corrected = torch.as_tensor(corrected, device=image.device, dtype=image.dtype)
        field = torch.as_tensor(field, device=image.device, dtype=image.dtype)
    return (corrected, field) if return_field else corrected
