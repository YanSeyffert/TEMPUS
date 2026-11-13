# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempus.model import _pairwise_sa


def _pairwise_sa_dense_reference(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    _, b, _ = X.shape
    dX = X[:, :, None, :] - X[:, None, :, :]        # (N, B, B, 3)
    rBA = np.linalg.norm(dX, axis=3)                # (N, B, B)
    diag = np.eye(b, dtype=bool)[None, :, :]
    rBA = np.where(diag, np.inf, rBA)
    return np.sum(mu[None, None, :] / rBA, axis=2)  # (N, B)


def test_pairwise_sa_matches_dense_reference():
    rng = np.random.default_rng(12345)
    n, b = 8, 9

    X = rng.normal(size=(n, b, 3)) * 1.0e11
    X += np.arange(b, dtype=float)[None, :, None] * 1.0e7
    mu = rng.uniform(1.0e8, 1.0e12, size=b)

    got = _pairwise_sa(X, mu)
    ref = _pairwise_sa_dense_reference(X, mu)
    np.testing.assert_allclose(got, ref, rtol=1.0e-12, atol=0.0)


def test_pairwise_sa_single_body_is_zero():
    X = np.zeros((5, 1, 3), dtype=np.float64)
    mu = np.array([1.23], dtype=np.float64)
    got = _pairwise_sa(X, mu)
    assert got.shape == (5, 1)
    np.testing.assert_allclose(got, 0.0, rtol=0.0, atol=0.0)
