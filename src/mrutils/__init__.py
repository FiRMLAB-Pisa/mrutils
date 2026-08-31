"""Thin MRI pre- and post-processing utilities.

Everything here is array in, array out. Torch tensors keep their device and
NumPy arrays stay NumPy, because the same step runs on a scanner host and on a
workstation and neither should have to convert to reach it.

The package is deliberately small. It holds what more than one reconstruction
package needs and nothing that belongs to a reconstruction itself: the centered
Fourier convention, the receive-array corrections that happen before a solve,
the EPI readout corrections, apodization, and the shading correction that
happens after.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._bias import bias_field_correct
from ._coils import coil_compress, noise_prewhiten
from ._epi import correct_lines, epi_ramp_operator, estimate_epi_phase
from ._fieldmap import field_map
from ._fourier import (
    centered_fftn,
    fftc,
    ifftc,
    resize_centered,
    resize_centered_axis,
    torch_or_numpy,
)
from ._readout import remove_readout_oversampling
from ._windows import apodize, fermi_window, hann_window

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "apodize",
    "bias_field_correct",
    "centered_fftn",
    "coil_compress",
    "correct_lines",
    "epi_ramp_operator",
    "estimate_epi_phase",
    "fermi_window",
    "fftc",
    "field_map",
    "hann_window",
    "ifftc",
    "noise_prewhiten",
    "remove_readout_oversampling",
    "resize_centered",
    "resize_centered_axis",
    "torch_or_numpy",
]
