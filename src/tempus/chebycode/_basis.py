# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
_basis.py — Chebyshev basis machinery: M⁻¹ cache, argument reduction, Clenshaw evaluation.

Three responsibilities, kept together because they all touch the same mathematics:

  1. get_basis(ncoeff, dist)
       Build and cache the inverse Vandermonde matrix M⁻¹ for a given
       (ncoeff, dist) pair.  Called once per combination; reused for every
       granule that shares those parameters.

  2. tau_from_jd(jds, t_start, t_end)
       Map raw JD values to the normalised coordinate τ ∈ [-1, +1] without
       catastrophic cancellation.  Even though the user supplies a single
       float64 JD (no split required), we split internally into integer +
       fractional parts so that the large (~2.4x10⁶) integer portions cancel
       exactly in integer arithmetic.

  3. clenshaw_batch / clenshaw_deriv_batch
       Evaluate the Chebyshev series (and its derivative) at M query points
       simultaneously using numpy broadcasting — this is the "parallel
       evaluation" path that replaces the scalar Clenshaw loop of chebycode.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._nodes import _nodes_xi


# ===========================================================================
# 1. Precomputed inverse Vandermonde
# ===========================================================================

@lru_cache(maxsize=64)
def get_basis(ncoeff: int, dist: int) -> np.ndarray:
    """
    Return M⁻¹ (float64, shape (ncoeff, ncoeff)) for (ncoeff, dist).

    Computed once per unique (ncoeff, dist) pair via SVD; all subsequent
    calls return the cached result.

    For dist=1 the Vandermonde matrix is modified before inversion: rows 1
    and ncoeff-1 are replaced by the Chebyshev derivative vectors evaluated
    at the corresponding node positions.  This encodes the C¹ boundary
    conditions (matching value+slope at shared granule endpoints).
    """
    xi = _nodes_xi(ncoeff, dist)
    M  = _vandermonde(xi, ncoeff)

    if dist == 1:
        # Row 1       carries the derivative BC at xi[1]       (near right edge).
        # Row ncoeff-1 carries the derivative BC at xi[ncoeff-1] (left endpoint).
        M[1,          :] = _cheb_deriv_row(xi[1],          ncoeff)
        M[ncoeff - 1, :] = _cheb_deriv_row(xi[ncoeff - 1], ncoeff)

    return _invert_svd(M)


# ===========================================================================
# 2. Argument reduction:  JD  →  τ ∈ [-1, +1]
# ===========================================================================

def tau_from_jd(
    jd_int:  np.ndarray,
    jd_frac: np.ndarray,
    t_start: float,
    t_end:   float,
) -> np.ndarray:
    """
    Map split JD values to τ ∈ [-1, +1] with cancellation-free arithmetic.

    τ = (2·JD - t_start - t_end) / (t_end - t_start)

    The numerator involves catastrophic cancellation when JD ≈ 2.4 x 10⁶ and
    the granule is only a few days wide.  By accepting jd_int / jd_frac
    separately (and re-splitting t_start / t_end internally), the large
    integer parts cancel exactly in integer arithmetic and only the small
    remainders contribute floating-point rounding:

        num_int  = 2·jd_int  - floor(t_start) - floor(t_end)   (exact, int64)
        num_frac = 2·jd_frac - frac(t_start)  - frac(t_end)    (small float)

    The denominator (t_end - t_start) = granule_len is already small; no
    splitting needed there.
    """
    jd_int  = np.asarray(jd_int,  dtype=np.int64)
    jd_frac = np.asarray(jd_frac, dtype=np.float64)

    ts_int  = int(np.floor(t_start));  ts_frac = t_start - ts_int
    te_int  = int(np.floor(t_end));    te_frac = t_end   - te_int

    num_int  = (2 * jd_int - ts_int - te_int).astype(np.float64)
    num_frac = 2.0 * jd_frac - ts_frac - te_frac

    return (num_int + num_frac) / (t_end - t_start)


# ===========================================================================
# 3. Vectorised Clenshaw evaluation
# ===========================================================================

def clenshaw_batch(coeffs: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Evaluate Σ c_j T_j(τ) for M query points simultaneously (backward recurrence).

    This is the "parallel evaluation" path: instead of looping over τ values,
    numpy broadcasting propagates the recurrence across the whole (M,) or
    (M, ndim) result array in one pass.

    Parameters
    ----------
    coeffs : (ncoeff,) or (ncoeff, ndim) Chebyshev coefficients
    taus   : (M,) query points in [-1, +1]

    Returns
    -------
    (M,) or (M, ndim) float64
    """
    multidim = coeffs.ndim > 1
    if multidim:
        t  = taus[:, np.newaxis]                        # (M, 1)  →  broadcasts vs (ndim,)
        b1 = np.zeros((len(taus), coeffs.shape[1]))     # (M, ndim)
        b2 = np.zeros_like(b1)
    else:
        t  = taus                                        # (M,)
        b1 = np.zeros(len(taus))
        b2 = np.zeros_like(b1)

    # Backward recurrence:  b_j = 2τ b_{j+1} - b_{j+2} + c_j
    for j in range(len(coeffs) - 1, 0, -1):
        b0 = 2.0 * t * b1 - b2 + coeffs[j]
        b2, b1 = b1, b0

    # Final step for T_0 = 1 (not 2τ):  result = τ b_1 - b_2 + c_0
    return t * b1 - b2 + coeffs[0]


def clenshaw_deriv_batch(coeffs: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Evaluate dS/dτ = Σ c_j T'_j(τ) for M query points simultaneously.

    Uses the forward recurrence for T_j and T'_j together:
        T_{j+1}  = 2τ T_j  - T_{j-1}
        T'_{j+1} = 2 T_j + 2τ T'_j - T'_{j-1}

    Returns dS/dτ.  Caller must multiply by dτ/dJD = 2 / granule_len to
    convert to physical units dS/dJD.

    Parameters
    ----------
    coeffs : (ncoeff,) or (ncoeff, ndim)
    taus   : (M,)

    Returns
    -------
    (M,) or (M, ndim) float64
    """
    ncoeff   = len(coeffs)
    multidim = coeffs.ndim > 1

    if multidim:
        M, D     = len(taus), coeffs.shape[1]
        t        = taus[:, np.newaxis]                  # (M, 1)
        T_prev   = np.ones((M, D))                      # T_0 = 1
        T_curr   = np.broadcast_to(t, (M, D)).copy()    # T_1 = τ
        dT_prev  = np.zeros((M, D))                     # T'_0 = 0
        dT_curr  = np.ones((M, D))                      # T'_1 = 1
    else:
        t        = taus
        T_prev   = np.ones(len(taus))
        T_curr   = t.copy()
        dT_prev  = np.zeros(len(taus))
        dT_curr  = np.ones(len(taus))

    # Accumulate: dS/dτ = Σ c_j · T'_j(τ)
    # T'_0 = 0  →  contributes nothing; T'_1 = 1  →  contributes c_1.
    result = np.zeros_like(T_prev)
    if ncoeff > 1:
        result = result + coeffs[1] * dT_curr

    for j in range(2, ncoeff):
        T_next   = 2.0 * t * T_curr  - T_prev
        dT_next  = 2.0 * T_curr + 2.0 * t * dT_curr - dT_prev
        result   = result + coeffs[j] * dT_next
        T_prev,  T_curr  = T_curr,  T_next
        dT_prev, dT_curr = dT_curr, dT_next

    return result


# ===========================================================================
# Private helpers
#
# _vandermonde and _cheb_deriv_row use np.longdouble (80-bit extended on
# x86 Linux/Mac; same as float64 on Windows).  T_n(x) near x = ±1 at high
# order accumulates rounding error during the recurrence, so extra precision
# during matrix construction reduces errors before the SVD inversion.
# Results are downcast to float64 in _invert_svd.
# ===========================================================================

def _vandermonde(xi: np.ndarray, ncoeff: int) -> np.ndarray:
    """Build the Chebyshev Vandermonde matrix M[i, j] = T_j(xi[i]) in longdouble."""
    xi = np.asarray(xi, dtype=np.longdouble)
    M  = np.ones((ncoeff, ncoeff), dtype=np.longdouble)
    if ncoeff > 1:
        M[:, 1] = xi
    for j in range(2, ncoeff):
        M[:, j] = 2.0 * xi * M[:, j - 1] - M[:, j - 2]
    return M


def _cheb_deriv_row(x: float, ncoeff: int) -> np.ndarray:
    """Row vector [T'_0(x), T'_1(x), ..., T'_{ncoeff-1}(x)] in longdouble."""
    x  = np.longdouble(x)
    T  = np.ones(ncoeff,  dtype=np.longdouble)
    dT = np.zeros(ncoeff, dtype=np.longdouble)
    if ncoeff > 1:
        T[1]  = x
        dT[1] = np.longdouble(1.0)
    for j in range(2, ncoeff):
        T[j]  = 2.0 * x * T[j - 1]  - T[j - 2]
        dT[j] = 2.0 * T[j - 1] + 2.0 * x * dT[j - 1] - dT[j - 2]
    return dT


def _invert_svd(M: np.ndarray) -> np.ndarray:
    """Invert via SVD (float64); M was built in longdouble for better conditioning."""
    M64 = np.asarray(M, dtype=np.float64)
    U, s, Vt = np.linalg.svd(M64)
    s_inv = np.where(s > 1e-14 * s[0], 1.0 / s, 0.0)
    return np.asarray((Vt.T * s_inv) @ U.T, dtype=np.float64)
