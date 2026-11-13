# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/integrate.py
# Numerical integrators for TEMPUS (cumulative integrals with I(x0)=0).

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional
from functools import lru_cache
import numpy as np

IntegratorName = Literal["trapz", "am4", "am8", "am10", "am12", "amk", "rk4"]


def _as_1d(a: np.ndarray) -> np.ndarray:
    # Preserve input dtype (allows longdouble experiments).
    return np.asarray(a).reshape(-1)


def _require_uniform_grid(x: np.ndarray, *, atol_step: float | None = None) -> float:
    """
    Ensure x has (approximately) constant spacing. Return step h.
    """
    x = _as_1d(x)
    if x.size < 2:
        return x.dtype.type(0.0) if x.dtype.kind == "f" else 0.0
    dx = np.diff(x)
    h = dx[0]
    if atol_step is None:
        # 1e-12 * h would be too tight for JD-derived ET arrays: float64 rounding
        # of et values far from J2000 introduces step noise ~ ulp(et) ~ et * 2.2e-16,
        # which can exceed 1e-12 * h for long spans.  Use 1e-9 * h as the tolerance
        # (still catches genuine step-size changes by a factor > 1e-9).
        atol_step = 1e-9 * max(1.0, abs(float(h)))
    if not np.allclose(dx, h, rtol=0.0, atol=atol_step):
        raise ValueError("This integrator requires a uniform grid (constant step size).")
    return h


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoid integral with I(x0)=0.
    """
    y = _as_1d(y)
    x = _as_1d(x)
    if y.size != x.size:
        raise ValueError("cumulative_trapezoid: y and x must have the same length.")
    n = x.size
    if n == 0:
        return np.array([], dtype=np.result_type(x, y))
    if n == 1:
        return np.array([0.0], dtype=np.result_type(x, y))

    dx = np.diff(x)
    inc = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty(n, dtype=np.result_type(x, y))
    out[0] = out.dtype.type(0.0)
    out[1:] = np.cumsum(inc)
    return out


def cumulative_trapezoid_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Fast cumulative trapezoid for uniform step h.
    """
    y = _as_1d(y)
    n = y.size
    if n == 0:
        return np.array([], dtype=y.dtype)
    if n == 1:
        return np.array([0.0], dtype=y.dtype)

    inc = 0.5 * h * (y[1:] + y[:-1])
    out = np.empty(n, dtype=y.dtype)
    out[0] = out.dtype.type(0.0)
    out[1:] = np.cumsum(inc)
    return out


def cumulative_am4_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Cumulative integral using fixed-step Adams-Moulton 4 (AM4), order 4 (uniform step).

        I_{n+1} = I_n + (h/24)*( 9 y_{n+1} + 19 y_n - 5 y_{n-1} + y_{n-2} )

    Warm-start with trapezoid for I[1..3].
    """
    y = _as_1d(y)
    n = y.size
    if n == 0:
        return np.array([], dtype=y.dtype)
    if n == 1:
        return np.array([0.0], dtype=y.dtype)
    if n < 4:
        return cumulative_trapezoid_uniform(y, h)

    out = np.empty(n, dtype=y.dtype)
    out[0] = out.dtype.type(0.0)
    out[1] = out[0] + 0.5 * h * (y[0] + y[1])
    out[2] = out[1] + 0.5 * h * (y[1] + y[2])
    out[3] = out[2] + 0.5 * h * (y[2] + y[3])

    h24 = h / 24.0
    yy = y
    I = out
    for k in range(3, n - 1):
        I[k + 1] = I[k] + h24 * (9.0 * yy[k + 1] + 19.0 * yy[k] - 5.0 * yy[k - 1] + yy[k - 2])
    return out


@lru_cache(maxsize=None)
def _adams_moulton_coeffs(k: int) -> np.ndarray:
    """
    Compute Adams-Moulton coefficients for k-step method (order k+1).

    Coefficients correspond to nodes at [0, -1, -2, ..., -k] and integrate
    the interpolant over [-1, 0].
    """
    if k < 1:
        raise ValueError("k must be >= 1 for Adams-Moulton coefficients.")

    x = np.arange(0, -k - 1, -1, dtype=np.float64)
    coeffs = []
    for j in range(k + 1):
        p = np.poly1d([1.0])
        denom = 1.0
        xj = x[j]
        for m in range(k + 1):
            if m == j:
                continue
            p = np.polymul(p, np.poly1d([1.0, -x[m]]))
            denom *= (xj - x[m])
        p = p / denom
        pint = np.polyint(p)
        bj = float(pint(0.0) - pint(-1.0))
        coeffs.append(bj)
    return np.asarray(coeffs, dtype=np.float64)


def cumulative_amk_uniform(y: np.ndarray, h: float, k: int) -> np.ndarray:
    """
    Cumulative integral using fixed-step Adams-Moulton k-step (order k+1).

    I_{n+1} = I_n + h * sum_{j=0..k} b_j * y_{n+1-j}
    """
    y = _as_1d(y)
    n = y.size
    if n == 0:
        return np.array([], dtype=y.dtype)
    if n == 1:
        return np.array([0.0], dtype=y.dtype)
    if n < k + 1:
        return cumulative_trapezoid_uniform(y, h)

    coeffs = _adams_moulton_coeffs(k)
    out = np.empty(n, dtype=y.dtype)
    out[0] = out.dtype.type(0.0)
    for i in range(1, k + 1):
        out[i] = out[i - 1] + 0.5 * h * (y[i - 1] + y[i])

    idx = np.arange(k + 1)
    for n_idx in range(k, n - 1):
        y_window = y[n_idx + 1 - idx]
        out[n_idx + 1] = out[n_idx] + h * float(np.dot(coeffs, y_window))
    return out


def cumulative_am10_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Cumulative integral using fixed-step Adams-Moulton 10 (AM10), order 10 (uniform step).
    """
    return cumulative_amk_uniform(y, h, k=9)

def cumulative_am12_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Cumulative integral using fixed-step Adams-Moulton 12 (AM12), order 12 (uniform step).
    """
    return cumulative_amk_uniform(y, h, k=11)


def cumulative_am8_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Cumulative integral using fixed-step Adams-Moulton 8 (AM8), order 8 (uniform step).
    """
    return cumulative_amk_uniform(y, h, k=7)


# rk4 == trapezoid mathematically, and rk4 is kept as an alias for trapz
def get_integrator(name: IntegratorName = "trapz", *, amk_k: int | None = None,) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a function integ(y, x) -> cumulative integral with I(x0)=0.
    """
    if name in ("trapz", "rk4"):
        return cumulative_trapezoid
    if name == "am4":
        def _am4(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            h = _require_uniform_grid(x)
            return cumulative_am4_uniform(y, h)
        return _am4
    if name == "am10":
        def _am10(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            h = _require_uniform_grid(x)
            return cumulative_am10_uniform(y, h)
        return _am10
    if name == "am12":
        def _am12(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            h = _require_uniform_grid(x)
            return cumulative_am12_uniform(y, h)
        return _am12
    if name == "am8":
        def _am8(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            h = _require_uniform_grid(x)
            return cumulative_am8_uniform(y, h)
        return _am8
    if name == "amk":
        if amk_k is None:
            raise ValueError("Integrator 'amk' requires amk_k (k-step order).")
        if amk_k < 1:
            raise ValueError("amk_k must be >= 1.")
        def _amk(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            h = _require_uniform_grid(x)
            return cumulative_amk_uniform(y, h, k=amk_k)
        return _amk
    raise ValueError(
        "Unknown integrator '{name}'. Supported: 'trapz', 'rk4', 'am4', 'am8', 'am10', 'am12', 'amk'."
        .format(name=name)
    )


@dataclass(frozen=True)
class PreparedGrid:
    """
    Cache uniform step-size verification once. Use for repeated integrals on the same x grid.
    """
    x: np.ndarray
    atol_step: Optional[float] = None

    def __post_init__(self):
        x = _as_1d(self.x)
        object.__setattr__(self, "x", x)
        h = _require_uniform_grid(x, atol_step=self.atol_step)
        object.__setattr__(self, "h", h)

    def integrate(self, y: np.ndarray, method: IntegratorName = "trapz") -> np.ndarray:
        if method in ("trapz", "rk4"):
            return cumulative_trapezoid_uniform(y, self.h)
        if method == "am4":
            return cumulative_am4_uniform(y, self.h)
        if method == "am10":
            return cumulative_am10_uniform(y, self.h)
        if method == "am12":
            return cumulative_am12_uniform(y, self.h)
        if method == "am8":
            return cumulative_am8_uniform(y, self.h)
        raise ValueError(f"Unknown method '{method}'.")
