# Examples

One notebook per application, each with the figure that checks it. Every
notebook opens in Colab, installs the package it needs, and runs top to bottom
on synthetic data — there is nothing to download.

| notebook | shows | checked against |
|---|---|---|
| [`fourier.ipynb`](fourier.ipynb) | `fftc`, `ifftc`, `resize_centered`, `remove_readout_oversampling` | the round trip, to machine precision |
| [`coils.ipynb`](coils.ipynb) | `noise_prewhiten`, `coil_compress` | the noise covariance becoming the identity |
| [`epi.ipynb`](epi.ipynb) | `estimate_epi_phase`, `correct_lines`, `epi_ramp_operator` | the impressed phase, and the ghost it leaves |
| [`apodization.ipynb`](apodization.ipynb) | `fermi_window`, `hann_window`, `apodize` | side-lobe height against main-lobe width |
| [`field_map.ipynb`](field_map.ipynb) | `field_map` | a known field, and where it aliases |
| [`bias_field.ipynb`](bias_field.ipynb) | `bias_field_correct` | a known shading, and one tissue's uniformity |
| [`density.ipynb`](density.ipynb) | `pipe_menon_dcf` | the closed form for radial, `1 / \|k\|` |

The figures under [`figures/`](figures/) are what the README shows, and each is
written by the notebook of the same name. Regenerating them is running the
notebooks:

```bash
pip install -e .[all] jupytext nbclient
jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb
```
