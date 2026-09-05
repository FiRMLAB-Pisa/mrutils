# %% [markdown]
# # The README's showcase figure
#
# One column per family of corrections, the step before it on top and after it
# below. Every panel is produced by the same calls the numbered examples make,
# on synthetic data, so it can be rebuilt anywhere:
#
# ```bash
# python examples/figures/make_showcase.py
# ```

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mrutils as mru

N = 128
rng = np.random.default_rng(0)


def phantom(n=N):
    """A Shepp-Logan phantom on an n x n grid."""
    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    ellipses = [
        (1.0, 0.0, 0.0, 0.69, 0.92, 0),
        (-0.8, 0.0, -0.0184, 0.6624, 0.874, 0),
        (-0.2, 0.22, 0.0, 0.11, 0.31, -18),
        (-0.2, -0.22, 0.0, 0.16, 0.41, 18),
        (0.1, 0.0, 0.35, 0.21, 0.25, 0),
        (0.1, 0.0, 0.1, 0.046, 0.046, 0),
        (0.1, 0.0, -0.1, 0.046, 0.046, 0),
        (0.1, -0.08, -0.605, 0.046, 0.023, 0),
        (0.1, 0.0, -0.606, 0.023, 0.023, 0),
        (0.1, 0.06, -0.605, 0.023, 0.046, 0),
    ]
    image = np.zeros((n, n))
    for value, cx, cy, a, b, degrees in ellipses:
        angle = np.deg2rad(degrees)
        xr = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        yr = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        image[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += value
    return image


# %% [markdown]
# ## The five columns

# %%
truth = phantom()
kspace = mru.fftc(truth)


def fourier():
    """The centred transform, and what comes back through it."""
    return (
        (np.log10(np.abs(kspace) + 1e-6), "magma", "log |fftc|"),
        (np.abs(mru.ifftc(kspace)), "gray", "ifftc(fftc(x))"),
    )


def coils(n_coils=8):
    """A receive array's noise covariance, and the same after prewhitening."""
    coupling = np.linalg.cholesky(
        0.35 ** np.abs(np.subtract.outer(np.arange(n_coils), np.arange(n_coils)))
        + 0.02 * np.eye(n_coils)
    )

    def coloured(shape):
        white = rng.normal(size=(n_coils, *shape)) + 1j * rng.normal(
            size=(n_coils, *shape)
        )
        return np.tensordot(coupling, white / np.sqrt(2), axes=(1, 0))

    noise = coloured((4096,))
    whitened = mru.noise_prewhiten(noise, noise, coil_axis=0)

    def covariance(data):
        flat = data.reshape(data.shape[0], -1)
        return np.abs(flat @ flat.conj().T / flat.shape[1])

    return (
        (covariance(noise), "viridis", "noise covariance"),
        (covariance(whitened), "viridis", "prewhitened"),
    )


def epi():
    """The Nyquist ghost an odd/even phase leaves, and its correction."""
    hybrid = mru.ifftc(kspace, axes=-1)
    readout = np.linspace(-0.5, 0.5, N)
    slope, offset = 0.9, 0.35
    train = []
    for ky in range(N):
        backwards = ky % 2 == 1
        sign = -1.0 if backwards else 1.0
        row = hybrid[ky] * np.exp(1j * sign * (slope * readout + offset))
        line = mru.fftc(row[None], axes=-1)
        train.append((line[:, ::-1] if backwards else line, backwards))

    # Three blip-nulled lines of alternating polarity see the same object, so
    # what separates them is the odd/even phase and nothing else.
    centre = hybrid[N // 2]
    forward = mru.fftc(
        (centre * np.exp(1j * (slope * readout + offset)))[None], axes=-1
    )
    backward = mru.fftc(
        (centre * np.exp(-1j * (slope * readout + offset)))[None], axes=-1
    )
    fit = mru.estimate_epi_phase([forward, backward, forward])

    def reconstruct(lines):
        return np.abs(mru.ifftc(np.concatenate(lines, axis=0)))

    return (
        (reconstruct(mru.correct_lines(train)), "gray", "Nyquist ghost"),
        (reconstruct(mru.correct_lines(train, fit)), "gray", "corrected"),
    )


def apodization():
    """Gibbs ringing off a truncated k-space, and a Fermi window against it."""
    truncated = mru.resize_centered(kspace, (48, 48))
    return (
        (np.abs(mru.ifftc(truncated)), "gray", "truncated k-space"),
        (np.abs(mru.ifftc(mru.apodize(truncated, kind="fermi"))), "gray", "apodized"),
    )


def bias_field():
    """A surface array's shading, and N4 against it."""
    y, x = np.mgrid[-1 : 1 : N * 1j, -1 : 1 : N * 1j]
    field = 0.55 + 1.1 * np.exp(-((x - 0.55) ** 2 + (y + 0.35) ** 2) / 0.9)
    field /= field.mean()
    observed = (truth * field + 0.01 * rng.normal(size=truth.shape)).astype(np.float32)
    # N4 needs to be told where the object is: on a phantom against pure
    # background, an unmasked estimate fits the air as well and comes back
    # worse than it started -- 0.62 against 0.27 across one tissue class.
    corrected = mru.bias_field_correct(observed, mask=(truth > 0.05).astype(np.uint8))
    tissue = np.abs(truth - 0.2) < 0.01
    for name, image in (("shaded", observed), ("N4", corrected)):
        values = image[tissue]
        print(f"  {name:7} spread across one tissue {values.std() / values.mean():.3f}")
    # Windowed on the tissue rather than the shell, which is four times
    # brighter and would leave the gradient invisible.
    window = 0.45
    return (
        (np.clip(observed, 0, window), "gray", "shaded"),
        (
            np.clip(corrected * observed.mean() / corrected.mean(), 0, window),
            "gray",
            "N4 corrected",
        ),
    )


# %% [markdown]
# ## The figure

# %%
COLUMNS = [
    ("centred transform", fourier),
    ("array prewhitening", coils),
    ("EPI ghost", epi),
    ("apodization", apodization),
    ("receive shading", bias_field),
]

figure, axes = plt.subplots(2, len(COLUMNS), figsize=(2.15 * len(COLUMNS), 4.9))
for column, (title, build) in enumerate(COLUMNS):
    for row, (image, colour, label) in enumerate(build()):
        panel = axes[row, column]
        panel.imshow(image, cmap=colour)
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_ylabel(label, fontsize=7.5)
    axes[0, column].set_title(title, fontsize=9)
figure.tight_layout()

destination = Path(__file__).parent if "__file__" in dir() else Path("examples/figures")
figure.savefig(destination / "showcase.png", dpi=150, bbox_inches="tight")
print("wrote", destination / "showcase.png")
