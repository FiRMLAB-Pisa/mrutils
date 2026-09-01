# mrutils

The MRI pre- and post-processing steps that every reconstruction needs and none
of them owns: the centred Fourier convention, the receive-array corrections
that happen before a solve, the EPI readout corrections, and the shading and
apodization that happen after.

[![Tests](https://github.com/FiRMLAB-Pisa/mrutils/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/mrutils/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/mrutils/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/mrutils)
[![PyPI](https://img.shields.io/pypi/v/mrutils.svg)](https://pypi.org/project/mrutils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Everything is array in, array out: Torch tensors keep their device and NumPy
arrays stay NumPy, so the same step runs on a scanner host and on a
workstation without either having to convert to reach it. Torch is an extra,
not a dependency.

The package is deliberately small, and it is the base of a family of MRI
packages rather than a reconstruction of its own — no solvers, no physics
operators, no sensitivity estimation.

## Quick Start

```bash
pip install mrutils                # numpy + scipy
pip install mrutils[torch]         # keep tensors on their device
pip install mrutils[bias]          # N4 bias-field correction
pip install mrutils[all]
```

The Fourier convention is `ifftshift -> fft(norm="ortho") -> fftshift`, stated
once so nothing downstream re-derives the shifts, and cropping and padding are
about the same centre so neither moves the object.

```python
import mrutils as mru

# the centred transform, over the last two axes by default
kspace = mru.fftc(image)
plane = mru.ifftc(volume, axes=-1)

# crop or pad about the centre, and drop readout oversampling
coarse = mru.resize_centered(kspace, (128, 128))
readout = mru.remove_readout_oversampling(digitised, 256)

# whiten against a noise-only scan, so the array's coupling stops being a term
whitened = mru.noise_prewhiten(kspace, noise_scan, coil_axis=0)

# compress the array, keeping the basis for every acquisition that follows
compressed, basis = mru.coil_compress(whitened, 8)  # or 0.95 of the energy
compressed = mru.apply_coil_compression(basis, later_acquisition)

# stream a scan too large for the card, off the host, for the same answer
compressed = mru.apply_coil_compression(basis, host_scan, device="cuda")

# the EPI Nyquist ghost, measured from a blip-nulled navigator
phase = mru.estimate_epi_phase([plus, minus, plus])
lines = mru.correct_lines(train, phase)  # [(data, is_reversed), ...]

# ramp sampling, as a change of basis rather than an interpolation
operator = mru.epi_ramp_operator(sampled_k, uniform_k, support=matrix)
on_grid = readout @ operator.T

# apodize: radial for isotropic resolution, separable to keep the corners
apodized = mru.apodize(kspace, kind="fermi")
slab = mru.apodize(volume, kind="hann", geometry="separable", axes=(-3, -2, -1))
window = mru.fermi_window((256, 256), radius=0.9, width=0.05)

# N4 on the coil-combined magnitude, after the solve
corrected = mru.bias_field_correct(magnitude)  # needs [bias]
```

## Examples

Each runs its function against something known: a round trip, a covariance, a
phase impressed on purpose, a field with a closed form.

| | | |
|---|---|---|
| [`fourier.ipynb`](examples/fourier.ipynb) | the centred convention, and what cropping about the wrong centre does | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/mrutils/blob/main/examples/fourier.ipynb) |
| [`coils.ipynb`](examples/coils.ipynb) | prewhitening a receive array, and compressing it | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/mrutils/blob/main/examples/coils.ipynb) |
| [`epi.ipynb`](examples/epi.ipynb) | the Nyquist ghost, and ramp sampling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/mrutils/blob/main/examples/epi.ipynb) |
| [`apodization.ipynb`](examples/apodization.ipynb) | windows against Gibbs ringing, radial against separable | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/mrutils/blob/main/examples/apodization.ipynb) |
| [`bias_field.ipynb`](examples/bias_field.ipynb) | N4 against the shading a surface array leaves | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/mrutils/blob/main/examples/bias_field.ipynb) |

## Related Works

- **ismrmrd-python-tools** —
  <https://github.com/ismrmrd/ismrmrd-python-tools>. The same idea: the
  reconstruction steps that are common to everyone, as plain array functions.
  This package keeps that scope and drops the ISMRMRD file format, so nothing
  here needs a data model to be called.
- **SimpleITK** — <https://simpleitk.org/>. Its
  `N4BiasFieldCorrectionImageFilter` is what `bias_field_correct` calls.
- Roemer PB, Edelstein WA, Hayes CE, Souza SP, Mueller OM. *The NMR phased
  array.* Magn Reson Med 1990;16:192-225. The noise covariance and the
  whitening built from it.
- Buehrer M, Pruessmann KP, Boesiger P, Kozerke S. *Array compression for MRI
  with large coil arrays.* Magn Reson Med 2007;57:1131-1139.
- Bruder H, Fischer H, Reinfelder HE, Schmitt F. *Image reconstruction for echo
  planar imaging with nonequidistant k-space sampling.* Magn Reson Med
  1992;23:311-323. The odd/even phase correction and the ramp-sampling
  resampling.
- Bernstein MA, Fain SB, Riederer SJ. *Effect of windowing and zero-filled
  reconstruction of MRI data on spatial resolution and acquisition strategy.*
  J Magn Reson Imaging 2001;14:270-280. The window kernels, the half-height-at-
  Nyquist normalization, and the radial-versus-separable geometry — whose
  52.4% (2D) and 50.7% (3D) corner ratios the examples reproduce.
- Tustison NJ, Avants BB, Cook PA, Zheng Y, Egan A, Yushkevich PA, Gee JC.
  *N4ITK: improved N3 bias correction.* IEEE Trans Med Imaging
  2010;29:1310-1320.

## Development

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh
pytest -q
```

The docstring examples run as part of the suite — they are the documentation,
and an example that has drifted is a broken one. See
[CONTRIBUTING.md](CONTRIBUTING.md).
