# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
_stencil.py — Interpolate raw samples to Chebyshev collocation nodes.

The Vandermonde system M·c = f requires function values at the node positions,
but integration output lands at arbitrary times.  For each node a local window
of npoints surrounding samples is selected and interpolated.

Two strategies depending on available data:

  With dy_raw (Hermite) — KroghInterpolator with interleaved repeated abscissas
      encodes both value and derivative at each sample point, producing a
      degree-2n-1 polynomial through n (value, derivative) pairs.

  Without dy_raw (Lagrange) — BarycentricInterpolator fits the values only.
      When a derivative is also needed (dist=1 boundary nodes), a second
      KroghInterpolator pass on non-repeated nodes provides it.

All times are relative to the granule start (0..granule_len days) to avoid
the large Julian date base (~2.4 x 10⁶) causing cancellation in divided differences.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BarycentricInterpolator, KroghInterpolator


def eval_at_nodes(
    t_nodes: np.ndarray,
    t_raw:   np.ndarray,
    y_raw:   np.ndarray,
    dy_raw:  np.ndarray | None,
    npoints: int,
    dist:    int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Interpolate raw samples to each collocation node.

    Parameters
    ----------
    t_nodes : (ncoeff,) node times in [0, granule_len], same units as t_raw
    t_raw   : (N,) sample times relative to granule start, sorted ascending
    y_raw   : (N,) or (N, ndim) sample values
    dy_raw  : (N,) or (N, ndim) derivatives dy/dt, or None
    npoints : number of raw samples used per node (stencil width)
    dist    : node distribution (1-3); dist=1 also returns derivatives at the
              two boundary nodes for the C¹ Vandermonde rows

    Returns
    -------
    f_nodes  : (ncoeff,) or (ncoeff, ndim) interpolated values at each node
    df_nodes : (ncoeff,) or (ncoeff, ndim) derivatives at nodes 1 and ncoeff-1;
               None when dist != 1
    """
    ncoeff = len(t_nodes)

    # Promote 1-D arrays to 2-D so the rest of the code is uniform.
    scalar = y_raw.ndim == 1
    if scalar:
        y_raw = y_raw[:, None]
    N, ndim = y_raw.shape

    if dy_raw is not None:
        dy_raw = np.asarray(dy_raw, dtype=np.float64)
        if dy_raw.ndim == 1:
            dy_raw = dy_raw[:, None]

    half_w = npoints // 2
    f_out  = np.empty((ncoeff, ndim), dtype=np.float64)
    df_out = np.empty((ncoeff, ndim), dtype=np.float64) if dist == 1 else None

    for j, t_j in enumerate(t_nodes):
        # Local stencil: npoints samples centred on the closest raw index,
        # clipped so the window never runs off either end of the array.
        ii = int(np.searchsorted(t_raw, t_j))
        ii = max(half_w, min(N - (npoints - half_w), ii))
        sl = slice(ii - half_w, ii - half_w + npoints)
        ta = t_raw[sl]

        # Boundary nodes 1 and ncoeff-1 need a derivative for the C¹ BC rows.
        need_deriv = (dist == 1) and (j == 1 or j == ncoeff - 1)

        if dy_raw is not None:
            # Hermite interpolation: interleave each abscissa with itself so
            # KroghInterpolator treats the duplicate as a derivative constraint.
            #   abscissas : [t0, t0, t1, t1, ...]
            #   values    : [y0, dy0, y1, dy1, ...]
            xs = np.repeat(ta, 2)
            ys = np.empty((2 * len(ta), ndim), dtype=np.float64)
            ys[0::2] = y_raw[sl]
            ys[1::2] = dy_raw[sl]
            interp   = KroghInterpolator(xs, ys)
            f_out[j] = interp(t_j)
            if need_deriv:
                df_out[j] = interp.derivative(t_j)
        else:
            # Lagrange interpolation via the numerically stable barycentric form.
            interp   = BarycentricInterpolator(ta, y_raw[sl])
            f_out[j] = interp(t_j)
            if need_deriv:
                # Derivative of the same polynomial, obtained via Krogh on non-repeated nodes.
                df_out[j] = KroghInterpolator(ta, y_raw[sl]).derivative(t_j)

    if scalar:
        f_out = f_out[:, 0]
        if df_out is not None:
            df_out = df_out[:, 0]

    return f_out, df_out

