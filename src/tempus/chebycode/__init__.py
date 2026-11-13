# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
chebycode — Piecewise Chebyshev fitting for time-series data.

Represents a time series as a sequence of short Chebyshev polynomial segments
(granules), each fitted independently.  Designed for high-precision ephemeris
data where Julian dates span millions of days but individual granules are only
a few days wide.

Julian dates
  All functions accept JD as two arrays: jd_int (int64) and jd_frac (float64).
  They are never added together inside the library — only file I/O merges them.

Node distributions (dist parameter)
  1  Gauss-Lobatto + C¹ boundary conditions — smoothest across granule borders,
     requires velocity data (dy).  [default]
  2  Gauss-Lobatto — endpoints included, no derivative constraint.
  3  Chebyshev-Gauss roots — interior nodes only, no endpoints.

Recommended settings
  Moon position      : granule_len=4,  order=12, dist=1  (requires dy=vel)
  Orientation angles : granule_len=8,  order=11, dist=2
  Scalar time series : granule_len=4,  order=11, dist=3

Quick start
-----------
    from tempus.chebycode import ChebyshevSeries
    from tempus.chebycode.validation import plot_compare_p
    import numpy as np

    # Fit
    cs = ChebyshevSeries.from_samples(
        jd_int, jd_frac, pos, dy=vel,
        granule_len=4, order=12, dist=1, n_workers=4,
    )
    cs.to_asc("moon.asc")

    # Load and evaluate
    cs  = ChebyshevSeries.from_asc("moon.asc")
    pos = cs.evaluate(jd_int, jd_frac)        # (N, ndim)
    vel = cs.evaluate_deriv(jd_int, jd_frac)  # (N, ndim)  [units: per day]

    # Diagnostics and validation
    cs.report()                                             # continuity + truncation summary
    cs.validate(jd_int, jd_frac, ref_pv, kind='position')  # error stats + residual plots

Public modules
--------------
  series.py     — ChebyshevSeries (main class)
  io.py         — save_ascii / load_ascii
  validation.py — numerical diagnostics + matplotlib validation plots

Private modules (implementation details, prefixed with _)
----------------------------------------------------------
  _nodes.py   — collocation node placement
  _stencil.py — local interpolation of nodes
  _basis.py   — M⁻¹ cache, tau_from_jd, clenshaw_batch
  _granule.py — Granule dataclass, fit_granule, evaluate_batch
"""

from .series import ChebyshevSeries
from .validation import (
    compare_against_samples,
    continuity_report,
    jd_to_yr,
    plot_compare_angle,
    plot_compare_angledot,
    plot_compare_p,
    plot_compare_v,
    truncation_error_estimate,
)

# aliases
ChebyshevSeries = ChebyshevSeries
ChebySeries     = ChebyshevSeries

__all__ = [
    "ChebyshevSeries",
    "ChebySeries",
    # diagnostics
    "compare_against_samples",
    "continuity_report",
    "truncation_error_estimate",
    # plotting
    "jd_to_yr",
    "plot_compare_p",
    "plot_compare_v",
    "plot_compare_angle",
    "plot_compare_angledot",
]
