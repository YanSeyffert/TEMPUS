# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import tempfile
import os

from tempus.chebycode import ChebyshevSeries, compare_against_samples, continuity_report
from tempus.chebycode._basis import clenshaw_batch, clenshaw_deriv_batch
from tempus.chebycode._nodes import _nodes_xi

_F = np.float64
JD0 = 2460000.5


def _split(jd: np.ndarray):
    """Split float64 JD array into (int64, float64) parts."""
    ji = np.floor(jd).astype(np.int64)
    jf = (jd - ji.astype(np.float64)).astype(np.float64)
    return ji, jf


def _eval1(cs, jd_scalar: float):
    """Evaluate series at a single scalar JD (merged float)."""
    ji = int(np.floor(jd_scalar))
    jf = float(jd_scalar - ji)
    return cs.evaluate(ji, jf)


def _eval1_deriv(cs, jd_scalar: float):
    ji = int(np.floor(jd_scalar))
    jf = float(jd_scalar - ji)
    return cs.evaluate_deriv(ji, jf)


# ===== dtype checks =====

def test_dtype_nodes_returns_float64():
    assert _nodes_xi(10, 2).dtype == np.float64


def test_dtype_clenshaw_batch_returns_float64():
    c = np.array([1.0, 2.0, 3.0], dtype=_F)
    tau = np.linspace(-1, 1, 5, dtype=_F)
    assert clenshaw_batch(c, tau).dtype == _F


def test_dtype_clenshaw_deriv_batch_returns_float64():
    c = np.array([1.0, 2.0, 3.0], dtype=_F)
    tau = np.linspace(-1, 1, 5, dtype=_F)
    assert clenshaw_deriv_batch(c, tau).dtype == _F


# ===== Chebyshev fundamentals =====

def test_cheb_Tn_cos_theta():
    """T_n(cos θ) = cos(n θ): verify clenshaw_batch with unit basis vectors."""
    theta = np.linspace(0, np.pi, 50, dtype=_F)
    tau = np.cos(theta)
    for n in range(11):
        c = np.zeros(n + 1, dtype=_F)
        c[n] = 1.0
        assert_allclose(clenshaw_batch(c, tau), np.cos(n * theta), atol=1e-12)


def test_cheb_clenshaw_deriv_vs_numerical():
    rng = np.random.default_rng(99)
    c = rng.standard_normal(8).astype(_F)
    tau = np.array([0.0, 0.3, -0.7], dtype=_F)
    eps = 1e-7
    num = (clenshaw_batch(c, tau + eps) - clenshaw_batch(c, tau - eps)) / (2 * eps)
    assert_allclose(clenshaw_deriv_batch(c, tau), num, atol=1e-5)


def test_cheb_multidim_clenshaw():
    rng = np.random.default_rng(7)
    c = rng.standard_normal((6, 3)).astype(_F)
    tau = np.linspace(-1, 1, 20, dtype=_F)
    result = clenshaw_batch(c, tau)
    assert result.shape == (20, 3)
    for d in range(3):
        assert_allclose(result[:, d], clenshaw_batch(c[:, d], tau), atol=1e-11)


# ===== from_samples with DENSE data (0.1 day spacing, plenty of points) =====

def test_dense_scalar_sin_granule_len4_order12():
    jd = np.linspace(JD0, JD0 + 40, 401, dtype=_F)
    y = np.sin(2 * np.pi * (jd - JD0) / 20.0).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, dist=3)
    assert cs.order == 12
    assert cs.granule_edges.dtype == _F
    assert_allclose(cs.evaluate(*_split(jd)), y, atol=1e-7)


def test_dense_scalar_sin_with_derivative_high_accuracy():
    jd = np.linspace(JD0, JD0 + 40, 401, dtype=_F)
    omega = 2 * np.pi / 20.0
    y = np.sin(omega * (jd - JD0)).astype(_F)
    dy = (omega * np.cos(omega * (jd - JD0))).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, dy=dy)
    jd_test = np.linspace(JD0 + 0.5, JD0 + 39.5, 500, dtype=_F)
    assert_allclose(cs.evaluate(*_split(jd_test)), np.sin(omega * (jd_test - JD0)), atol=1e-8)


def test_dense_evaluate_scalar_returns_scalar_for_scalar_input():
    jd = np.linspace(JD0, JD0 + 10, 101, dtype=_F)
    y = (jd - JD0).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=5, dist=3)
    val = _eval1(cs, JD0 + 2.0)
    assert np.ndim(val) == 0


def test_dense_evaluate_deriv_matches_true_derivative():
    jd = np.linspace(JD0, JD0 + 20, 201, dtype=_F)
    omega = 2 * np.pi / 10.0
    y = np.sin(omega * (jd - JD0)).astype(_F)
    dy = (omega * np.cos(omega * (jd - JD0))).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, dy=dy, granule_len=4, order=12)
    jd_test = np.linspace(JD0 + 0.5, JD0 + 19.5, 100, dtype=_F)
    dy_test = omega * np.cos(omega * (jd_test - JD0))
    assert_allclose(cs.evaluate_deriv(*_split(jd_test)), dy_test, atol=1e-5)


# ===== Granule boundary continuity =====

def test_boundary_position_continuous_across_edges():
    jd = np.linspace(JD0, JD0 + 30, 301, dtype=_F)
    y = (np.exp(-0.05 * (jd - JD0)) * np.sin(0.5 * (jd - JD0))).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=10, dist=2)
    eps = 1e-10
    for edge in cs.granule_edges[1:-1]:
        y_l = _eval1(cs, float(edge) - eps)
        y_r = _eval1(cs, float(edge) + eps)
        assert abs(y_l - y_r) < 1e-6, f"pos jump: {abs(y_l - y_r)}"


def test_boundary_with_derivative_tighter_continuity():
    jd = np.linspace(JD0, JD0 + 30, 301, dtype=_F)
    y = (np.exp(-0.05 * (jd - JD0)) * np.sin(0.5 * (jd - JD0))).astype(_F)
    dy = (np.exp(-0.05 * (jd - JD0)) * (
        -0.05 * np.sin(0.5 * (jd - JD0)) + 0.5 * np.cos(0.5 * (jd - JD0))
    )).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, dy=dy, granule_len=4, order=10)
    eps = 1e-10
    for edge in cs.granule_edges[1:-1]:
        y_l = _eval1(cs, float(edge) - eps)
        y_r = _eval1(cs, float(edge) + eps)
        assert abs(y_l - y_r) < 1e-8, f"pos jump: {abs(y_l - y_r)}"
        v_l = _eval1_deriv(cs, float(edge) - eps)
        v_r = _eval1_deriv(cs, float(edge) + eps)
        assert abs(v_l - v_r) < 1e-3, f"vel jump: {abs(v_l - v_r)}"


# ===== 3D orbit (Moon-like) =====

def test_3d_moon_like_orbit():
    jd = np.linspace(JD0, JD0 + 30, 601, dtype=_F)
    omega = 2 * np.pi / 27.3
    pos = np.column_stack([
        384400 * np.cos(omega * (jd - JD0)),
        384400 * np.sin(omega * (jd - JD0)),
        20000 * np.sin(2 * omega * (jd - JD0)),
    ]).astype(_F)
    vel = np.column_stack([
        -384400 * omega * np.sin(omega * (jd - JD0)),
         384400 * omega * np.cos(omega * (jd - JD0)),
         20000 * 2 * omega * np.cos(2 * omega * (jd - JD0)),
    ]).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), pos, dy=vel, granule_len=4, order=13)
    assert cs.ndim == 3
    jd_test = np.linspace(JD0 + 0.5, JD0 + 29.5, 200, dtype=_F)
    pos_fit = cs.evaluate(*_split(jd_test))
    pos_true = np.column_stack([
        384400 * np.cos(omega * (jd_test - JD0)),
        384400 * np.sin(omega * (jd_test - JD0)),
        20000 * np.sin(2 * omega * (jd_test - JD0)),
    ])
    assert_allclose(pos_fit, pos_true, atol=0.01)


# ===== Orientation angles =====

def test_orientation_angles():
    jd = np.linspace(JD0, JD0 + 50, 501, dtype=_F)
    angles = np.column_stack([
        0.01 * (jd - JD0) + 0.001 * np.sin(0.1 * (jd - JD0)),
        0.5 + 0.002 * np.cos(0.05 * (jd - JD0)),
    ]).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), angles, granule_len=8, order=10, dist=2)
    assert cs.ndim == 2
    jd_test = np.linspace(JD0 + 1, JD0 + 49, 300, dtype=_F)
    fit = cs.evaluate(*_split(jd_test))
    true_val = np.column_stack([
        0.01 * (jd_test - JD0) + 0.001 * np.sin(0.1 * (jd_test - JD0)),
        0.5 + 0.002 * np.cos(0.05 * (jd_test - JD0)),
    ])
    assert_allclose(fit, true_val, atol=1e-7)


# ===== Sparse data =====

def test_sparse_075_day_spacing():
    jd = np.arange(JD0, JD0 + 20.01, 0.75, dtype=_F)
    y = np.sin(0.3 * (jd - JD0)).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=12, dist=3)
    jd_test = np.linspace(cs.jd_start + 0.1, cs.jd_end - 0.1, 200, dtype=_F)
    assert_allclose(cs.evaluate(*_split(jd_test)), np.sin(0.3 * (jd_test - JD0)), atol=2e-2)


def test_sparse_2_day_spacing():
    jd = np.arange(JD0, JD0 + 30.01, 2.0, dtype=_F)
    y = np.cos(0.2 * (jd - JD0)).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=8, dist=3)
    jd_test = np.linspace(cs.jd_start + 0.5, cs.jd_end - 0.5, 100, dtype=_F)
    assert_allclose(cs.evaluate(*_split(jd_test)), np.cos(0.2 * (jd_test - JD0)), atol=5e-2)


# ===== ASCII serialisation =====

def test_ascii_round_trip_preserves_coefficients():
    jd = np.linspace(JD0, JD0 + 16, 161, dtype=_F)
    y = np.sin(0.5 * (jd - JD0)).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=8, dist=3)
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False, mode="w") as f:
        path = f.name
    try:
        cs.to_asc(path)
        cs2 = ChebyshevSeries.from_asc(path)
        assert cs2.order == cs.order
        assert_allclose(cs2.granule_edges, cs.granule_edges, atol=1e-14)
        for c1, c2 in zip(cs.coefficients, cs2.coefficients):
            assert_allclose(c1, c2, atol=1e-14)
    finally:
        os.unlink(path)


def test_ascii_file_header_starts_with_ChebyshevSeries():
    jd = np.linspace(JD0, JD0 + 8, 81, dtype=_F)
    y = (jd - JD0).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=4, dist=3)
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False, mode="w") as f:
        path = f.name
    try:
        cs.to_asc(path)
        with open(path) as fh:
            assert fh.readline().strip().startswith("ChebyshevSeries")
    finally:
        os.unlink(path)


def test_ascii_multidim_round_trip():
    jd = np.linspace(JD0, JD0 + 20, 201, dtype=_F)
    y = np.column_stack([
        np.sin(0.2 * (jd - JD0)),
        np.cos(0.2 * (jd - JD0)),
        0.01 * (jd - JD0),
    ]).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=10, dist=3)
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False, mode="w") as f:
        path = f.name
    try:
        cs.to_ascii(path)
        cs2 = ChebyshevSeries.from_ascii(path)
        jd_test = np.linspace(cs.jd_start + 0.5, cs.jd_end - 0.5, 80, dtype=_F)
        assert_allclose(cs.evaluate(*_split(jd_test)), cs2.evaluate(*_split(jd_test)), atol=1e-13)
    finally:
        os.unlink(path)


# ===== Edge cases =====

def test_edge_single_granule():
    jd = np.linspace(JD0, JD0 + 3, 31, dtype=_F)
    y = ((jd - JD0) ** 2).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=10, order=5, dist=3)
    assert cs.n_granules == 1
    assert_allclose(_eval1(cs, JD0 + 1.5), 1.5**2, atol=1e-10)


def test_edge_exact_polynomial_recovery():
    jd = np.linspace(JD0, JD0 + 8, 161, dtype=_F)
    t = jd - JD0
    y = (3.0 * t**3 - 2.0 * t**2 + t - 7.0).astype(_F)
    dy = (9.0 * t**2 - 4.0 * t + 1.0).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, dy=dy, granule_len=4, order=6)
    jd_test = np.linspace(cs.jd_start + 0.01, cs.jd_end - 0.01, 200, dtype=_F)
    t_test = jd_test - JD0
    y_true = 3.0 * t_test**3 - 2.0 * t_test**2 + t_test - 7.0
    assert_allclose(cs.evaluate(*_split(jd_test)), y_true, atol=1e-6)


def test_edge_repr_is_informative():
    jd = np.linspace(JD0, JD0 + 10, 101, dtype=_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), np.sin(jd - JD0).astype(_F), dist=3)
    r = repr(cs)
    assert "ChebyshevSeries" in r
    assert "order=12" in r
    assert "ndim=1" in r


# ===== TCG-TT style =====

def test_tcgtt_small_perturbation_on_large_jd_baseline():
    jd = np.linspace(2460000.5, 2460100.5, 1001, dtype=_F)
    y = (65.5e-6 * (jd - 2460000.5) + 1.5e-9 * np.sin(
        2 * np.pi * (jd - 2460000.5) / 10.0
    )).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=12, dist=3)
    jd_test = np.linspace(2460001.0, 2460100.0, 500, dtype=_F)
    y_true = 65.5e-6 * (jd_test - 2460000.5) + 1.5e-9 * np.sin(
        2 * np.pi * (jd_test - 2460000.5) / 10.0
    )
    residual = np.max(np.abs(cs.evaluate(*_split(jd_test)) - y_true))
    assert residual < 1e-12, f"residual = {residual}"


# ===== Diagnostics =====

def test_continuity_report_keys():
    jd = np.linspace(JD0, JD0 + 12, 241, dtype=_F)
    y = np.sin(0.2 * (jd - JD0)).astype(_F)
    cs = ChebyshevSeries.from_samples(*_split(jd), y, granule_len=4, order=8, dist=3)

    rep = continuity_report(cs)
    assert "max_abs_jump" in rep
    assert "rms_jump" in rep
    assert "max_abs_deriv_jump" in rep
    assert "rms_deriv_jump" in rep


def test_compare_against_samples():
    jd = np.linspace(JD0, JD0 + 12, 241, dtype=_F)
    y = np.column_stack([
        np.sin(0.2 * (jd - JD0)),
        np.cos(0.2 * (jd - JD0)),
    ]).astype(_F)
    ji, jf = _split(jd)
    cs = ChebyshevSeries.from_samples(ji, jf, y, granule_len=4, order=10, dist=3)

    rep = compare_against_samples(cs, ji, jf, y)
    assert rep["max_abs_error"] < 1e-8
    assert "rms_error" in rep
    assert "mean_error" in rep
