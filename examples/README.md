# Examples

One example per family, each with the figure that checks it. The `.py` is the
source: it runs as a script, lints with the rest of the package, and reads as a
diff. The `.ipynb` beside it is generated from it, executed, and committed with
its outputs, so it opens in Colab, installs what it needs and shows its figures
without being run.

| example | shows | checked against |
|---|---|---|
| [`01-fourier`](01-fourier.ipynb) | `fftc`, `ifftc`, `resize_centered`, `remove_readout_oversampling` | the round trip, to machine precision |
| [`02-coils`](02-coils.ipynb) | `noise_prewhiten`, `coil_compress`, `apply_coil_compression` | the noise covariance becoming the identity, and the device peak when a basis is applied |
| [`03-epi`](03-epi.ipynb) | `estimate_epi_phase`, `correct_lines`, `epi_ramp_operator` | the impressed phase, and the ghost it leaves |
| [`04-apodization`](04-apodization.ipynb) | `fermi_window`, `hann_window`, `apodize` | side-lobe height against main-lobe width |
| [`05-bias_field`](05-bias_field.ipynb) | `bias_field_correct` | a known shading, and one tissue's uniformity |

[`figures/make_showcase.py`](figures/make_showcase.py) draws the README's
figure from the same calls, and is not one of the examples.

## Rebuilding

```bash
pip install -e .[all] jupytext nbclient ipykernel
bash scripts/build_examples.sh
```

Every notebook is regenerated from its script and executed against the
interpreter the package is installed into, which is also what writes the
figures under [`figures/`](figures/). `--check` verifies the notebooks are
current without running them.
