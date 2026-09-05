# %% [markdown]
# # Apodization
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/mrutils/blob/main/examples/04-apodization.ipynb)
#
# `fermi_window`, `hann_window` and `apodize`: trading resolution for a point
# spread function without side lobes.

# %%
try:
    import mrutils  # noqa: F401
except ImportError:
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mrutils[torch]", "matplotlib"],
        check=True,
    )

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mrutils as mru


def phantom(n=128):
    """A Shepp-Logan phantom on an n x n grid."""
    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    ellipses = [
        # value, centre x, centre y, semi-axis a, semi-axis b, degrees
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


# The object is reconstructed on a 128 grid from a 64-wide measurement, which
# is what makes the truncation -- and its ringing -- real rather than nominal.
grid, acquired = 128, 64
image = phantom(grid).astype(complex)
measured = mru.resize_centered(mru.fftc(image), (acquired, acquired))
print(f"acquired {measured.shape}, reconstructed on {image.shape}")

# %% [markdown]
# ## Two windows
#
# A reconstruction that transforms a sharply truncated k-space gets the
# truncation's ringing with it. An apodization window rolls the measurement off
# towards its edge instead, trading resolution for a point spread function
# without side lobes.
#
# `fermi_window` sets its radius and its transition width separately, so the
# passband stays wide while the roll-off is gentle. `hann_window` is the raised
# cosine over the whole radius: it starts tapering at the origin, so it costs
# more resolution and leaves less ringing. The table below is that trade, in
# numbers.
#
# The point spread functions are evaluated on a grid eight times finer than the
# image, and that is not cosmetic: an unwindowed measurement's side lobes fall
# exactly on the image grid's sample points, so a PSF read off the image grid
# looks side-lobe free when it is nothing of the sort.

# %%
fermi = mru.fermi_window((acquired, acquired), radius=0.8, width=0.08)
hann = mru.hann_window((acquired, acquired))
flat = np.ones((acquired, acquired))

upsample = 8


def psf(window):
    """The window's point spread function, on a grid ``upsample`` times finer."""
    fine = mru.resize_centered(
        window.astype(complex), (upsample * acquired, upsample * acquired)
    )
    profile = np.abs(mru.ifftc(fine))[upsample * acquired // 2]
    return profile / profile.max()


offset = (np.arange(upsample * acquired) - upsample * acquired // 2) / upsample

print(f"{'':>6}  {'side lobe':>10}  {'FWHM':>8}")
for label, window in (("none", flat), ("fermi", fermi), ("hann", hann)):
    profile = psf(window)
    lobe = profile[np.abs(offset) > 3].max()
    half = np.flatnonzero(profile > 0.5)
    print(f"{label:>6}  {lobe:>10.1e}  {offset[half[-1]] - offset[half[0]]:>6.2f} px")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].plot(
    np.linspace(-1, 1, acquired), fermi[acquired // 2], label="fermi(0.8, 0.08)"
)
axes[0].plot(np.linspace(-1, 1, acquired), hann[acquired // 2], label="hann")
axes[0].set_title("window, central profile")
axes[0].set_xlabel("k / Nyquist")
axes[0].legend(fontsize=8)

for label, window in (("none", flat), ("fermi", fermi), ("hann", hann)):
    axes[1].semilogy(offset, psf(window) + 1e-12, label=label)
axes[1].set_xlim(-10, 10)
axes[1].set_ylim(1e-4, 2)
axes[1].set_title("point spread function")
axes[1].set_xlabel("pixels")
axes[1].legend(fontsize=8)

shown = axes[2].imshow(fermi, cmap="viridis")
axes[2].set_title("fermi window")
axes[2].set_xticks([])
axes[2].set_yticks([])
fig.colorbar(shown, ax=axes[2], fraction=0.046)
fig.tight_layout()

# %% [markdown]
# ## What it buys
#
# Gibbs ringing is the overshoot at a sharp edge, and more resolution does not
# remove it — it only makes it narrower. The profile below crosses the phantom's
# skull, which is the sharpest edge in the object.
#
# A window is not a blur: its point spread function has negative lobes, so an
# apodized image can still exceed the object's own maximum. What the numbers
# show is how much less it does so.


# %%
def reconstruct(kspace):
    return mru.ifftc(mru.resize_centered(kspace, (grid, grid)))


truncated = reconstruct(measured)
apodized = reconstruct(mru.apodize(measured, kind="fermi", radius=0.8, width=0.08))
tapered = reconstruct(mru.apodize(measured, kind="hann"))

estimates = (("no window", truncated), ("fermi", apodized), ("hann", tapered))
for label, estimate in estimates:
    over = np.abs(estimate).max() - np.abs(image).max()
    print(f"{label:>10}: overshoot above the object {over:+.3f}")

# %%
row = 96  # a row crossing the skull, the sharpest edge the object has
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for axis, (label, estimate) in zip(axes[:3], estimates, strict=True):
    axis.imshow(np.abs(estimate), cmap="gray", vmin=0, vmax=1.15)
    axis.axhline(row, color="tab:red", linewidth=0.8)
    axis.set_title(label)
    axis.set_xticks([])
    axis.set_yticks([])

axes[3].plot(np.abs(image)[row], color="k", linewidth=0.8, label="object")
for label, estimate in estimates:
    axes[3].plot(np.abs(estimate)[row], label=label)
axes[3].set_xlim(8, 48)
axes[3].set_title("across the edge")
axes[3].set_xlabel("pixel")
axes[3].legend(fontsize=8)
fig.tight_layout()

# %%
# Written only when the notebook runs inside a clone, and it is what the
# README shows -- so the figure in the README is always this cell's output.
figures = Path("figures")
if figures.is_dir():
    fig.savefig(figures / "apodization.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## Radial or separable
#
# The same one-dimensional kernel can be extended over the grid two ways. The
# *radial* geometry evaluates it on the Euclidean radius, giving an ellipsoid;
# the *separable* geometry multiplies it along each axis, which keeps more of
# k-space's corners. Bernstein et al. quantify the difference at the diagonal
# Nyquist point, and both of their numbers come back below.
#
# The trade is real: radial has the higher SNR and the more isotropic point
# spread function, separable the better diagonal resolution. Note that the
# radial window is exactly 0.5 where each axis meets Nyquist, which is what
# leaves the small residual lobes along the axes of its PSF.

# %%
# Bernstein et al. Eq. 13: at the diagonal Nyquist point the radial window
# admits 52.4% of what the separable one does in 2D, and 50.7% in 3D. The
# one-dimensional kernel at u is the window evaluated at DC with its radius
# moved to 1 - u, so both numbers come out of the public function.
transition = 10.0 / 128  # the paper's N = 256 setting


def kernel(u):
    return float(mru.fermi_window((4, 4), radius=1.0 - u, width=transition)[2, 2])


for dimensions in (2, 3):
    diagonal = kernel(1 / np.sqrt(dimensions)) ** dimensions
    print(
        f"{dimensions}D: radial admits {100 * 0.5 / diagonal:.1f}% of separable "
        f"at the diagonal Nyquist point"
    )

radial = mru.fermi_window((128, 128), geometry="radial")
separable = mru.fermi_window((128, 128), geometry="separable")
print(f"at the corner: radial {radial[0, 0]:.4f}, separable {separable[0, 0]:.4f}")


# %%
def psf_image(window, upsample=4):
    fine = mru.resize_centered(
        window.astype(complex), tuple(upsample * s for s in window.shape)
    )
    pattern = np.abs(mru.ifftc(fine))
    return pattern / pattern.max()


fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
for axis, window, title in (
    (axes[0], radial, "radial"),
    (axes[1], separable, "separable"),
):
    shown = axis.imshow(window, cmap="viridis", vmin=0, vmax=1)
    axis.set_title(f"{title} window")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(shown, ax=axis, fraction=0.046)

middle = 2 * 128
for axis, window, title in (
    (axes[2], radial, "radial"),
    (axes[3], separable, "separable"),
):
    pattern = psf_image(window)[middle - 20 : middle + 20, middle - 20 : middle + 20]
    axis.imshow(np.log10(pattern + 1e-6), cmap="magma", vmin=-4, vmax=0)
    axis.set_title(f"{title} PSF, log")
    axis.set_xticks([])
    axis.set_yticks([])
fig.tight_layout()

# %% [markdown]
# ## A slab
#
# Nothing about either geometry is two-dimensional. Each axis is normalized to
# its own Nyquist edge, so a slab gets an ellipsoid matched to its own grid and
# every axis reaches half height at its own Nyquist.

# %%
# A slab is the same window with a third axis, and `apodize` names them.
volume = np.ones((32, 64, 64), dtype=complex)
apodized_volume = mru.apodize(volume, kind="fermi", axes=(-3, -2, -1))
print(f"{volume.shape} -> {apodized_volume.shape}")

window = mru.fermi_window((32, 256, 256))
print(
    f"centre {window[16, 128, 128]:.4f}, "
    f"each axis edge {window[0, 128, 128]:.3f} {window[16, 0, 128]:.3f} "
    f"{window[16, 128, 0]:.3f}"
)

# The default transition is ten samples of the longest axis: one scalar cannot
# be ten samples of every axis, and taking it from the shortest would put the
# roll-off inside the passband. A slab that wants ten samples across its short
# axis asks for them.
print(
    f"default width {10 / (256 // 2):.4f}, "
    f"which is {10 / (256 // 2) * (32 // 2):.1f} samples on the 32 axis"
)

# %% [markdown]
# ## Anisotropic matrices
#
# Both windows are radial in *normalized* k, so each axis reaches its own
# Nyquist edge at 1. An anisotropic matrix therefore gets an ellipsoid matched
# to its own grid rather than a sphere that clips one axis long before the
# other.

# %%
wide = mru.fermi_window((64, 256))
print(f"a {wide.shape} matrix gets an ellipse, not a circle")

fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
shown = axes[0].imshow(wide, cmap="viridis", aspect="auto")
axes[0].set_title("fermi_window((64, 256))")
fig.colorbar(shown, ax=axes[0], fraction=0.02)
axes[1].plot(np.linspace(-1, 1, 64), wide[:, 128], label="short axis")
axes[1].plot(np.linspace(-1, 1, 256), wide[32], "--", label="long axis")
axes[1].set_xlabel("k / that axis's Nyquist")
axes[1].set_title("each axis rolls off at its own edge")
axes[1].legend(fontsize=8)
fig.tight_layout()
