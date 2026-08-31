# mrutils

Thin MRI pre- and post-processing utilities: centred FFTs, coil compression,
noise prewhitening, partial Fourier, EPI corrections, apodization and
bias-field correction.

[![Tests](https://github.com/FiRMLAB-Pisa/mrutils/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/mrutils/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/mrutils/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/mrutils)
[![PyPI](https://img.shields.io/pypi/v/mrutils.svg)](https://pypi.org/project/mrutils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Everything here is array in, array out. Torch tensors keep their device and
NumPy arrays stay NumPy, so the same step runs on a scanner host and on a
workstation without either having to convert to reach it.

This is the base of a family of small MRI packages, and it holds only what more
than one of them needs. Nothing here knows about a reconstruction: no solvers,
no physics operators, no sensitivity estimation.

## Install

```bash
pip install mrutils                # numpy + scipy
pip install mrutils[torch]         # keep tensors on their device
pip install mrutils[dcf]           # Pipe-Menon density compensation
pip install mrutils[bias]          # N4 bias-field correction
pip install mrutils[all]
```

Torch is deliberately not a hard dependency. Every function takes a NumPy array
as readily as a tensor, and a host that only ever sees NumPy should not have to
install a deep-learning runtime to centre an FFT.

## What is in it

### The Fourier convention

The one thing an MRI reconstruction means by "the Fourier transform":
`ifftshift -> fft(norm="ortho") -> fftshift`, stated once so nothing downstream
re-derives the shifts.

```python
import mrutils as mru

kspace = mru.fftc(image)  # last two axes by default
plane = mru.ifftc(volume, axes=-1)  # decouple a readout
padded = mru.resize_centered(kspace, (256, 256))
```

### Before the solve

```python
compressed, basis = mru.coil_compress(kspace, 8)  # or 0.95 of the energy
whitened = mru.noise_prewhiten(kspace, noise_scan, coil_axis=0)
cropped = mru.remove_readout_oversampling(readout, 256)
```

`coil_compress` returns the basis as well as the compressed data, so the
acquisitions that arrive after a calibration — and the sensitivities they are
solved against — are compressed the same way.

### Partial Fourier

An image whose phase varies slowly is nearly conjugate symmetric in k-space, so
an omitted edge is implied by the acquired one. `POCS` iterates towards an image
that reproduces every acquired sample; `Homodyne` reaches an answer in one pass.

```python
image = mru.fill_partial_echo(kspace, readout_mask, dimension=2)
image = mru.fill_partial_echo(kspace, readout_mask, dimension=2, method="homodyne")
```

### EPI

Ramp-sampling resampling is a change of basis rather than an interpolation: the
readout is band-limited, so samples taken anywhere determine it everywhere.

```python
operator = mru.epi_ramp_operator(sampled_k, uniform_k, support=matrix)
on_grid = train @ operator.T

phase = mru.estimate_epi_phase([plus, minus, plus])
corrected = mru.correct_lines(train_lines, phase)  # [(line, is_reversed), ...]
```

The fit is the *correction*, so it comes back as the negative of the phase the
reversed line was carrying.

### After the solve

```python
apodized = mru.apodize(kspace, kind="fermi", radius=0.9, width=0.05)
apodized = mru.apodize(kspace, kind="hann")
corrected = mru.bias_field_correct(magnitude)  # N4, needs [bias]
offset_hz = mru.field_map(echo_images, echo_times)
weights = mru.pipe_menon_dcf(trajectory, (256, 256))  # needs [dcf]
```

Windows are radial in normalized k, so an anisotropic matrix gets an ellipsoid
matched to its own grid rather than a sphere that clips one axis first.

## Development

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh
pytest -q
```

The docstring examples run as part of the suite — they are the documentation,
and an example that has drifted is a broken one. See
[CONTRIBUTING.md](CONTRIBUTING.md).
