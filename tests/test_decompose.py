"""
Minimal smoke tests for the GoDec decomposition.
Run with: pytest tests/
"""

import numpy as np
import pytest
from godec.decompose import _GreGoDec, _vectorize, _reconstruct


def make_synthetic_stack(n_frames=50, height=20, width=20, rank=2, seed=42):
    """Generate a synthetic low-rank + sparse + noise stack."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_frames, rank))
    V = rng.standard_normal((rank, height * width))
    L = U @ V
    S = rng.standard_normal((n_frames, height * width)) * 0.05
    S[S < 0.1] = 0  # make it sparse
    N = rng.standard_normal((n_frames, height * width)) * 0.01
    return (L + S + N), n_frames, height, width


def test_gredgodec_output_shape():
    D, n, h, w = make_synthetic_stack()
    D_out, L, S = _GreGoDec(D, rank=3, tau=0.5, tol=0.01, power=3, k=2)
    assert D_out.shape == D.shape
    assert L.shape == D.shape
    assert S.shape == D.shape


def test_gredgodec_residual():
    """The noise term should be small relative to the input."""
    D, n, h, w = make_synthetic_stack()
    D_out, L, S = _GreGoDec(D, rank=3, tau=0.5, tol=0.01, power=3, k=2)
    G = D_out - L - S
    assert np.linalg.norm(G) / np.linalg.norm(D_out) < 0.5


def test_vectorize_reconstruct_roundtrip():
    """Vectorize followed by manual reshape should recover original shape."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (10, 16, 16)).astype(float)
    flat = _vectorize(image, 10, 16, 16)
    assert flat.shape == (10, 256)
    for i in range(9):  # reconstruct uses range(time-1)
        recovered = flat[i].reshape((16, 16))
        np.testing.assert_array_equal(recovered, image[i])
