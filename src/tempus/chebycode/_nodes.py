# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
_nodes.py — Collocation node placement for Chebyshev granule fitting.

Nodes are ordered descending in ξ: xi[0] = +1 (right end of granule) down to
xi[ncoeff-1] = -1 (left end).  This ordering matches the cos(kπ/n) convention
and is required by _basis.py when it builds the Vandermonde matrix.
"""
from __future__ import annotations

import numpy as np


def place_nodes(ncoeff: int, dist: int, t_start: float, t_end: float) -> np.ndarray:
    """
    Return ncoeff collocation node times in [t_start, t_end].

    Parameters
    ----------
    ncoeff         : number of nodes (= polynomial order + 1)
    dist           : node distribution — see _nodes_xi for details
    t_start, t_end : physical interval endpoints (same time unit as the raw data)
    """
    xi = _nodes_xi(ncoeff, dist)
    # Affine map ξ ∈ [-1, +1]  →  t ∈ [t_start, t_end]
    #   t = 1/2 · ((t_end - t_start)·ξ + (t_start + t_end))
    return 0.5 * ((t_end - t_start) * xi + (t_start + t_end))


def _nodes_xi(ncoeff: int, dist: int) -> np.ndarray:
    """
    Return ncoeff node positions in [-1, +1], ordered from +1 toward -1.

    dist=1 — C¹ Lobatto
        Interior Gauss-Lobatto grid of ncoeff-2 nodes, with the outermost
        interior values (+1 and -1) copied to positions 0 and ncoeff-1:

            xi[0]        = xi[1]        = +1
            xi[1..ncoeff-2]  Gauss-Lobatto on [-1, +1]
            xi[ncoeff-1] = xi[ncoeff-2] = -1

        The duplicate endpoints mean that Vandermonde rows 1 and ncoeff-1
        (which _basis.py replaces with derivative rows) are evaluated at
        exactly ±1 — the granule boundary — enforcing C¹ continuity there.
        Requires ncoeff >= 4 (order >= 3).

    dist=2 — Gauss-Lobatto
        Standard cos(kπ/n) grid for k = 0..n, endpoints ±1 included.
        No derivative boundary conditions.

    dist=3 — Chebyshev-Gauss
        Roots of T_ncoeff: cos((2k+1)π / (2·ncoeff)) for k = 0..ncoeff-1.
        Interior nodes only; does not include ±1.
    """
    n = ncoeff - 1
    k = np.arange(ncoeff, dtype=float)

    if dist == 1:
        if ncoeff < 4:
            raise ValueError("dist=1 (C¹) requires order >= 3 (ncoeff >= 4)")
        ki = np.arange(ncoeff - 2, dtype=float)
        xi = np.empty(ncoeff)
        xi[1:ncoeff-1] = np.cos(ki * np.pi / (ncoeff - 3))
        xi[0]          = xi[1]        # = +1
        xi[ncoeff-1]   = xi[ncoeff-2] # = -1
        return xi
    elif dist == 2:
        return np.cos(k * np.pi / n)
    elif dist == 3:
        return np.cos((2 * k + 1) * np.pi / (2 * ncoeff))
    else:
        raise ValueError(f"dist must be 1, 2, or 3, got {dist}")
