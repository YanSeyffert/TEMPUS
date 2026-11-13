# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
series.py — ChebyshevSeries: the top-level user-facing class.

Fitting Cheb — pass n_workers > 1 to fit granules in parallel (uses threads).
Evaluating Cheb — evaluate() and evaluate_deriv() accept a full array of query JDs 
    and return all results in one call; no need for threading thanks to 
    numpy broadcasting inside clenshaw_batch().

Julian dates are passed as two arrays (jd_int, jd_frac) and kept separate
throughout to preserve full numerical precision.  Only file I/O merges them.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .validation import print_report, validate_series
from ._granule import Granule, evaluate_batch, evaluate_deriv_batch, fit_granule
from .io import load_ascii, save_ascii


@dataclass
class ChebyshevSeries:
    """Piecewise Chebyshev series Object Class for time-series data over a JD range."""

    granules: list[Granule]
    order:    int

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def jd_start(self) -> float:
        return self.granules[0].t_start

    @property
    def jd_end(self) -> float:
        return self.granules[-1].t_end

    @property
    def n_granules(self) -> int:
        return len(self.granules)

    @property
    def ndim(self) -> int:
        c = self.granules[0].coeffs
        return 1 if c.ndim == 1 else c.shape[1]

    @property
    def granule_edges(self) -> np.ndarray:
        edges = [g.t_start for g in self.granules]
        edges.append(self.granules[-1].t_end)
        return np.array(edges, dtype=np.float64)

    @property
    def coefficients(self) -> list[np.ndarray]:
        return [g.coeffs for g in self.granules]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        jd_int:      np.ndarray,
        jd_frac:     np.ndarray,
        y:           np.ndarray,
        dy:          np.ndarray | None = None,
        *,
        granule_len: float = 4.0,
        order:       int   = 12,
        npoints:     int   = 10,
        dist:        int   = 1,
        orientation: bool  = False,
        n_workers:   int   = 1,
    ) -> "ChebyshevSeries":
        """
        Fit a piecewise Chebyshev series from a sampled datapoints of a time-series.

        Parameters
        ----------
        jd_int      : (N,) integer parts of Julian dates  (e.g. 2451545)
        jd_frac     : (N,) fractional parts of Julian dates  (e.g. 0.5)
                      The split is preserved throughout to maintain full precision in the large JD values.
        y           : (N,) or (N, ndim) sample values
        dy          : (N,) or (N, ndim) derivatives dy/dJD (units in per day), or None
        granule_len : window length in days (default 4)
        order       : polynomial degree per granule (ncoeff = order+1)
        npoints     : stencil size for local interpolation (default 10)
        dist        : node distribution:
                        1  Gauss-Lobatto with C¹ boundary conditions (requires dy)  [default]
                        2  Gauss-Lobatto, endpoints included
                        3  Chebyshev-Gauss roots, interior only
        orientation : if True, unwrap angular data (removes 2π phase jumps)
        n_workers   : thread-pool size for parallel granule fitting (default 1).
                      In practice numpy's internal BLAS threads already saturate
                      the CPU, so values > 1 rarely improve wall time.
        """
        if dist == 1 and dy is None:
            raise ValueError("dist=1 (C¹) requires derivatives (dy). If not available, use dist=2 or dist=3.")

        jd_int  = np.asarray(jd_int,  dtype=np.int64)
        jd_frac = np.asarray(jd_frac, dtype=np.float64)
        y       = np.asarray(y,        dtype=np.float64)
        if dy is not None:
            dy = np.asarray(dy, dtype=np.float64)
        if orientation:
            y = np.unwrap(y, axis=0)

        ncoeff  = order + 1
        npoints = min(npoints, len(jd_int))

        # Span in days -- exact integer subtraction for the large base, small
        # float for the remainder.  Only used for n_gran; no merging needed.
        span   = float(jd_int[-1] - jd_int[0]) + (jd_frac[-1] - jd_frac[0])
        n_gran = max(1, int(np.ceil(span / granule_len)))

        # Build uniformly-spaced granule edges in split form.
        # Starting from the first sample's (int, frac), each successive edge
        # is offset by g * granule_len.  Carry arithmetic keeps frac in [0, 1).
        e0_int  = int(jd_int[0])
        e0_frac = float(jd_frac[0])
        edges_int  = np.empty(n_gran + 1, dtype=np.int64)
        edges_frac = np.empty(n_gran + 1, dtype=np.float64)
        for g in range(n_gran + 1):
            total_frac    = e0_frac + g * granule_len
            carry         = int(np.floor(total_frac))
            edges_int[g]  = e0_int + carry
            edges_frac[g] = total_frac - carry
        # Snap last edge exactly to the final data point.
        edges_int[-1]  = int(jd_int[-1])
        edges_frac[-1] = float(jd_frac[-1])

        # Relative combined JD for window searchsorted inside _fit.
        # Subtract the first sample's (int, frac) so all values are small
        # (0..span days), giving full float64 precision in the differences.
        jd_rel = (jd_int - e0_int).astype(np.float64) + (jd_frac - e0_frac)

        def _fit(g: int) -> Granule:
            gs_int_g  = int(edges_int[g]);   gs_frac_g = float(edges_frac[g])
            ge_int_g  = int(edges_int[g+1]); ge_frac_g = float(edges_frac[g+1])
            # Granule boundaries as combined floats (stored in Granule; re-split
            # in tau_from_jd with the same floor/frac pattern).
            gs = gs_int_g + gs_frac_g
            ge = ge_int_g + ge_frac_g
            # Relative edge positions (same reference as jd_rel) for searchsorted.
            gs_rel = float(gs_int_g - e0_int) + (gs_frac_g - e0_frac)
            ge_rel = float(ge_int_g - e0_int) + (ge_frac_g - e0_frac)
            # Window: npoints extra samples beyond each edge for the stencil.
            lo = max(0,          int(np.searchsorted(jd_rel, gs_rel)) - npoints)
            hi = min(len(jd_rel), int(np.searchsorted(jd_rel, ge_rel)) + npoints)
            y_w  = y[lo:hi]
            dy_w = dy[lo:hi] if dy is not None else None
            # Relative sample times in [0, granule_len]: exact integer difference
            # for the large base + small float for the remainder.
            t_raw = (jd_int[lo:hi] - gs_int_g).astype(np.float64) + (jd_frac[lo:hi] - gs_frac_g)
            return fit_granule(gs, ge, t_raw, y_w, dy_w, ncoeff, dist, npoints)

        if n_workers == 1:
            granules = [_fit(g) for g in range(n_gran)]
        else:
            # ThreadPoolExecutor: numpy and scipy release the GIL during heavy
            # array work, so threads genuinely run in parallel here.
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                granules = list(pool.map(_fit, range(n_gran)))

        return cls(granules=granules, order=order)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _granule_idx(self, jd_int: np.ndarray, jd_frac: np.ndarray) -> np.ndarray:
        """Return 0-based granule index for each JD value via binary search.

        Granules are days wide; float64 at JD~2.4e6 resolves to ~0.2 ms,
        so a plain merge is accurate enough for locating the correct granule.
        """
        jd = jd_int.astype(np.float64) + jd_frac
        idx = np.searchsorted(self.granule_edges, jd, side="right") - 1
        return np.clip(idx, 0, self.n_granules - 1)

    def _eval(self, jd_int, jd_frac, deriv: bool) -> np.ndarray:
        scalar    = np.ndim(jd_int) == 0
        jd_int_a  = np.atleast_1d(np.asarray(jd_int,  dtype=np.int64 )).ravel()
        jd_frac_a = np.atleast_1d(np.asarray(jd_frac, dtype=np.float64)).ravel()
        g_idx     = self._granule_idx(jd_int_a, jd_frac_a)

        result = (
            np.empty(len(jd_int_a))
            if self.ndim == 1
            else np.empty((len(jd_int_a), self.ndim))
        )

        fn = evaluate_deriv_batch if deriv else evaluate_batch
        # Each unique granule is evaluated as a batch — clenshaw_batch uses
        # numpy broadcasting over all query points in that granule at once.
        for g in np.unique(g_idx):
            mask         = g_idx == g
            result[mask] = fn(self.granules[g], jd_int_a[mask], jd_frac_a[mask])

        return result[0] if scalar else result

    def evaluate(self, jd_int, jd_frac) -> np.ndarray:
        """
        Evaluate the series at one or more JD values.

        Parameters
        ----------
        jd_int  : scalar or (M,) int64 integer parts of Julian dates
        jd_frac : scalar or (M,) float64 fractional parts of Julian dates

        Returns (M,) or (M, ndim) float64.
        """
        return self._eval(jd_int, jd_frac, deriv=False)

    def evaluate_deriv(self, jd_int, jd_frac) -> np.ndarray:
        """
        Evaluate d/dJD at one or more JD values.

        Returns (M,) or (M, ndim) float64.
        """
        return self._eval(jd_int, jd_frac, deriv=True)

    evaluate_jd = evaluate

    # ------------------------------------------------------------------
    # Diagnostics  (delegated to validation.py)
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a formatted continuity and truncation summary."""
        print_report(self)

    def validate(
        self,
        jd_int:        np.ndarray,
        jd_frac:       np.ndarray,
        pv_ref:        np.ndarray,
        *,
        kind:          str   = "position",
        label:         str   = "",
        x_tick:        float = 2,
        show_overview: bool  = False,
        unwrap:        bool  = False,
    ) -> dict:
        """
        Compare against reference samples and show residual plots.

        Parameters
        ----------
        jd_int, jd_frac : query times in split form
        pv_ref          : (N, 6) reference array — values in cols 0:3,
                          derivatives in cols 3:6
        kind            : 'position' or 'orientation'
        label           : plot title prefix
        x_tick          : x-axis tick spacing in years
        show_overview   : if True, also plot raw values alongside residuals
        unwrap          : unwrap angles before differencing (orientation only)

        Returns
        -------
        dict with keys max_abs_error, rms_error, mean_error
        """
        return validate_series(
            self, jd_int, jd_frac, pv_ref,
            kind=kind, label=label, x_tick=x_tick,
            show_overview=show_overview, unwrap=unwrap,
        )

    # ------------------------------------------------------------------
    # Serialisation  (delegated to io.py)
    # ------------------------------------------------------------------

    def to_asc(self, path: str, title: str = "") -> None:
        """Save to a human-readable ASCII file (readable by both packages)."""
        save_ascii(
            path,
            granule_edges=self.granule_edges,
            coefficients=self.coefficients,
            order=self.order,
            granule_len=float(self.granules[0].t_end - self.granules[0].t_start),
            title=title or "ChebyshevSeries",
        )

    to_ascii = to_asc

    @classmethod
    def from_asc(cls, path: str) -> "ChebyshevSeries":
        """Load from an ASCII file (ChebyshevSeries or INPOP format)."""
        data   = load_ascii(path)
        edges  = np.asarray(data["granule_edges"], dtype=np.float64)
        granules = [
            Granule(
                t_start=float(edges[g]),
                t_end=float(edges[g + 1]),
                coeffs=np.asarray(data["coefficients"][g], dtype=np.float64),
            )
            for g in range(len(edges) - 1)
        ]
        return cls(granules=granules, order=int(data["order"]))

    from_ascii = from_asc

    def __repr__(self) -> str:
        return (
            f"ChebyshevSeries(order={self.order}, n_granules={self.n_granules}, "
            f"ndim={self.ndim}, jd=[{self.jd_start:.1f}, {self.jd_end:.1f}])"
        )
