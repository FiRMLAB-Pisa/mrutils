"""Shared fixtures.

Every function in this package promises to keep a Torch tensor a Torch tensor,
on its own device, and a NumPy array a NumPy array. The ``to_array`` fixture is
how a test states that once and gets all three cases.
"""

import numpy as np
import pytest


def _torch_or_none():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_available():
    torch = _torch_or_none()
    return torch is not None and torch.cuda.is_available()


@pytest.fixture(
    params=[
        "numpy",
        pytest.param(
            "torch",
            marks=pytest.mark.skipif(_torch_or_none() is None, reason="no torch"),
        ),
        pytest.param(
            "torch-cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not _cuda_available(), reason="no CUDA device"),
            ],
        ),
    ]
)
def to_array(request):
    """Return a callable putting a NumPy array into the namespace under test."""
    if request.param == "numpy":
        return np.asarray

    import torch

    device = "cuda" if request.param == "torch-cuda" else "cpu"

    def convert(value):
        return torch.as_tensor(np.asarray(value), device=device)

    return convert


@pytest.fixture
def as_numpy():
    """Bring a result of any namespace back to NumPy for comparison."""

    def convert(value):
        detach = getattr(value, "detach", None)
        return value.detach().cpu().numpy() if detach else np.asarray(value)

    return convert


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not _cuda_available(), reason="no CUDA device"),
            ],
        ),
    ]
)
def device(request):
    """Run the test on each device this machine actually has."""
    return request.param
