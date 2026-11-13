# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
_granule.py — Single piecewise Chebyshev window: fitting and evaluation.

A Granule is the leaf unit of the piecewise representation.  It owns:
  - its time boundaries [t_start, t_end] (absolute JD)
  - its Chebyshev coefficients (ncoeff,) or (ncoeff, ndim)

fit_granule   builds one Granule from raw samples.
evaluate_batch / evaluate_deriv_batch query it at an array of JD values
and return results for all query points simultaneously (the vectorised path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._basis import clenshaw_batch, clenshaw_deriv_batch, get_basis, tau_from_jd
from ._nodes import place_nodes
from ._stencil import eval_at_nodes


@dataclass
class Granule:
    """One piecewise window: absolute JD boundaries and Chebyshev coefficients."""
    t_start: float
    t_end:   float
    coeffs:  np.ndarray   # (ncoeff,) for scalar data, (ncoeff, ndim) for vector


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_granule(
    t_start:  float,
    t_end:    float,
    t_raw:    np.ndarray,
    y_raw:    np.ndarray,
    dy_raw:   np.ndarray | None,
    ncoeff:   int,
    dist:     int,
    npoints:  int,
) -> Granule:
    """
    Fit one Chebyshev granule over [t_start, t_end].

    Parameters
    ----------
    t_start, t_end : absolute JD boundaries of this granule
    t_raw   : (N,) sample times **relative to t_start** (small numbers,
              no cancellation from large JD values)
    y_raw   : (N,) or (N, ndim) sample values
    dy_raw  : (N,) or (N, ndim) derivatives dy/dJD, or None → Lagrange only
    ncoeff  : number of Chebyshev coefficients (= order + 1)
    dist    : node distribution 1–4 (see _nodes.py)
    npoints : stencil width for local interpolation

    Steps
    -----
    1. Place ncoeff nodes in [0, granule_len]  (relative coordinates).
    2. Interpolate raw samples to each node (Hermite or Lagrange).
    3. coeffs = M⁻¹ · rhs   (M⁻¹ is retrieved from the lru_cache in _basis.py).
    """
    granule_len = t_end - t_start

    # Nodes in relative coordinates [0, granule_len] so the stencil sees
    # the same small numbers as t_raw.
    t_nodes = place_nodes(ncoeff, dist, 0.0, granule_len)

    f_nodes, df_nodes = eval_at_nodes(t_nodes, t_raw, y_raw, dy_raw, npoints, dist)

    M_inv = get_basis(ncoeff, dist)

    if dist == 1 and df_nodes is not None:
        # Rows 1 and ncoeff-1 of M carry derivative BCs.
        # Physical derivatives (d/dt) must be scaled to normalised units (d/dτ):
        #   d/dτ = d/dt · (granule_len / 2)
        half_len        = granule_len / 2.0
        rhs             = f_nodes.copy()
        rhs[1]          = df_nodes[1]          * half_len
        rhs[ncoeff - 1] = df_nodes[ncoeff - 1] * half_len
    else:
        rhs = f_nodes

    return Granule(t_start=t_start, t_end=t_end, coeffs=M_inv @ rhs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    granule:  Granule,
    jd_int:   np.ndarray,
    jd_frac:  np.ndarray,
) -> np.ndarray:
    """
    Evaluate the granule at multiple JD values simultaneously.

    jd_int / jd_frac must correspond to points within [granule.t_start,
    granule.t_end] (guaranteed by series.py via the granule index).

    Returns (M,) or (M, ndim) float64.
    """
    taus = tau_from_jd(jd_int, jd_frac, granule.t_start, granule.t_end)
    return clenshaw_batch(granule.coeffs, taus)


def evaluate_deriv_batch(
    granule:  Granule,
    jd_int:   np.ndarray,
    jd_frac:  np.ndarray,
) -> np.ndarray:
    """
    Evaluate d/dJD at multiple JD values simultaneously.

    clenshaw_deriv_batch returns dS/dτ; the chain rule scales to dS/dJD:
        dτ/dJD = 2 / granule_len

    Returns (M,) or (M, ndim) float64.
    """
    taus     = tau_from_jd(jd_int, jd_frac, granule.t_start, granule.t_end)
    dtau_djd = 2.0 / (granule.t_end - granule.t_start)
    return clenshaw_deriv_batch(granule.coeffs, taus) * dtau_djd
