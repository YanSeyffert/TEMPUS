# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
io.py — File I/O for ChebyshevSeries.

Three loaders are supported: the native ChebyshevSeries format, the INPOP native
.asc format, and a heuristic fallback for unrecognised external formats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# ASCII — save
# ---------------------------------------------------------------------------

def save_ascii(
    path,
    *,
    granule_edges,
    coefficients,
    order,
    granule_len: float | None = None,
    title: str = "ChebyshevSeries",
) -> None:
    """
    Save a piecewise Chebyshev series to a human-readable ASCII file.

    Format:
      Line 0  : title string (must start with "ChebyshevSeries")
      Line 1  : key=value metadata
      Line 2  : jd_start / jd_end (informational)
      Line 3  : column-header comment
      Data    : one row per (granule, dimension) —
                jd_start  jd_end  c0 … cN
                ndim consecutive rows share the same edge values.
    """
    path          = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    granule_edges = np.asarray(granule_edges, dtype=np.float64)
    n_granules    = len(coefficients)
    ndim          = 1 if np.asarray(coefficients[0]).ndim == 1 else int(coefficients[0].shape[1])

    if granule_len is None and len(granule_edges) >= 2:
        granule_len = float(granule_edges[1] - granule_edges[0])

    coeff_labels = "  ".join(f"c{i}" for i in range(order + 1))
    col_header   = "# jd_start  jd_end  " + coeff_labels

    if not title.startswith("ChebyshevSeries"):
        title = f"ChebyshevSeries — {title}" if title else "ChebyshevSeries"

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        meta = f"order={order}  ncoeff={order + 1}  ndim={ndim}  n_granules={n_granules}"
        if granule_len is not None:
            meta += f"  granule_len={granule_len:.10g}"
        f.write(meta + "\n")
        f.write(f"jd_start={granule_edges[0]:.6f}  jd_end={granule_edges[-1]:.6f}\n")
        f.write(col_header + "\n")

        for g, coeff in enumerate(coefficients):
            gs = float(granule_edges[g])
            ge = float(granule_edges[g + 1])
            edge_str = f"{gs:.17g}  {ge:.17g}"

            coeff_2d = np.asarray(coeff, dtype=np.float64)
            if coeff_2d.ndim == 1:
                coeff_2d = coeff_2d[:, None]   # (ncoeff, 1)
            # Write one row per dimension.
            for d in range(coeff_2d.shape[1]):
                coef_str = "  ".join(f"{coeff_2d[k, d]:+.17e}" for k in range(order + 1))
                f.write(f"{edge_str}  {coef_str}\n")


# ---------------------------------------------------------------------------
# ASCII — load dispatcher
# ---------------------------------------------------------------------------

def _parse_fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def load_ascii(path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")
    if lines[0].startswith("ChebyshevSeries"):
        return _load_chebyshev_fit(lines)
    if lines[0].lower().startswith("version"):
        return _load_inpop_asc(lines)
    return _load_external_asc(lines, path)


# ---------------------------------------------------------------------------
# Loader: ChebyshevSeries native format
# ---------------------------------------------------------------------------

def _load_chebyshev_fit(lines: list[str]) -> dict[str, object]:
    """Load the format written by save_ascii."""
    # Line 0: title (used for dispatch)
    # Line 1: key=value metadata
    meta: dict[str, str] = {}
    for token in lines[1].split():
        if "=" in token:
            k, v = token.split("=", 1)
            meta[k] = v

    order      = int(meta["order"])
    ndim       = int(meta["ndim"])
    n_granules = int(meta["n_granules"])

    # Skip header lines 0-2 and any #-comment lines
    data_lines = [l for l in lines[3:] if not l.startswith("#")]
    if len(data_lines) < n_granules * ndim:
        raise ValueError(
            f"Expected {n_granules * ndim} data lines, found {len(data_lines)}."
        )

    coefficients: list = []
    starts: list[float] = []
    ends:   list[float] = []

    ptr = 0
    for _ in range(n_granules):
        coeff_d: list[list[float]] = []
        for d in range(ndim):
            parts = data_lines[ptr].split()
            ptr  += 1
            if d == 0:
                starts.append(float(parts[0]))
                ends.append(float(parts[1]))
            coeff_d.append([_parse_fortran_float(p) for p in parts[2:]])

        if ndim == 1:
            coefficients.append(np.array(coeff_d[0], dtype=np.float64))
        else:
            # coeff_d is (ndim, ncoeff); stack to (ncoeff, ndim).
            coefficients.append(np.array(coeff_d, dtype=np.float64).T)

    granule_edges = np.array(starts + [ends[-1]], dtype=np.float64)
    return {
        "granule_edges": granule_edges, 
        "coefficients": coefficients, 
        "order": order
    }


# ---------------------------------------------------------------------------
# Loader: INPOP native .asc format
# ---------------------------------------------------------------------------

def _inpop_parse_ndim(meta_line: str) -> int:
    """
    Extract ndim from an INPOP metadata line such as:
      "Moon  Geocentric equator position  km  3  13  4  18720  ..."
      "TCB-TCG  Barycenter equator time  s  1  13  4  18719  ..."
    The first purely numeric token after the last alphabetic token is ndim.
    """
    parts     = meta_line.split()
    last_text = -1
    for i, tok in enumerate(parts):
        if any(c.isalpha() for c in tok):
            last_text = i
    try:
        return int(float(parts[last_text + 1]))
    except (ValueError, IndexError):
        return 1


def _load_inpop_asc(lines: list[str]) -> dict[str, object]:
    """
    Load an INPOP native .asc file.

    Format:
      Line 0 : "version :  YYYY.MMDD"
      Line 1 : metadata (name, units, ndim, order, dist_type, n_granules, ...)
      Lines 2+: data rows — jd_start  jd_end  c0  c1  ...  cN
                For ndim > 1, ndim consecutive rows share the same jd_start/jd_end.
    """
    if len(lines) < 3:
        raise ValueError("INPOP .asc file has too few lines.")

    ndim = _inpop_parse_ndim(lines[1])

    # Collect data rows: lines starting with two parseable floats
    data_rows: list[list[str]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            _parse_fortran_float(parts[0])
            _parse_fortran_float(parts[1])
        except ValueError:
            continue
        data_rows.append(parts)

    if not data_rows:
        raise ValueError("No data rows found in INPOP .asc file.")

    # Group into granules: ndim consecutive rows per granule
    n_complete = (len(data_rows) // ndim) * ndim
    coefficients: list = []
    starts: list[float] = []
    ends:   list[float] = []

    for i in range(0, n_complete, ndim):
        group = data_rows[i : i + ndim]
        jd_s = _parse_fortran_float(group[0][0])
        jd_e = _parse_fortran_float(group[0][1])
        starts.append(jd_s)
        ends.append(jd_e)

        coeff_d = [
            [_parse_fortran_float(tok) for tok in row[2:]]
            for row in group
        ]

        if ndim == 1:
            coefficients.append(np.array(coeff_d[0], dtype=np.float64))
        else:
            # coeff_d is (ndim, order+1); stack to (order+1, ndim)
            coefficients.append(np.array(coeff_d, dtype=np.float64).T)

    if not coefficients:
        raise ValueError("No complete granules found in INPOP .asc file.")

    ncoeff = coefficients[0].shape[0]
    granule_edges = np.array(starts + [ends[-1]], dtype=np.float64)
    return {
        "granule_edges": granule_edges,
        "coefficients": coefficients,
        "order": ncoeff - 1,
    }


# ---------------------------------------------------------------------------
# Loader: unrecognised external format (heuristic fallback, 1-D only)
# ---------------------------------------------------------------------------

def _load_external_asc(lines: list[str], path) -> dict[str, object]:
    starts:    list[float]       = []
    ends:      list[float]       = []
    coef_rows: list[list[float]] = []

    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            a = _parse_fortran_float(parts[0])
            b = _parse_fortran_float(parts[1])
        except ValueError:
            continue
        try:
            coeffs = [_parse_fortran_float(tok) for tok in parts[2:]]
        except ValueError as exc:
            raise ValueError(f"Invalid coefficient record in {path}: {line}") from exc
        if not coeffs:
            continue
        starts.append(a)
        ends.append(b)
        coef_rows.append(coeffs)

    if not starts:
        raise ValueError(f"No Chebyshev records found in {path}.")

    if len(starts) != len(set(starts)):
        raise ValueError(
            f"{path}: multiple data rows share the same jd_start — "
            "file does not appear to be a 1-D single-row-per-granule format."
        )

    order_idx  = np.argsort(starts)
    starts     = [starts[i]    for i in order_idx]
    ends       = [ends[i]      for i in order_idx]
    coef_rows  = [coef_rows[i] for i in order_idx]

    n_coeffs      = max(len(row) for row in coef_rows)
    granule_edges = np.array(starts + [ends[-1]], dtype=np.float64)
    coefficients  = []
    for row in coef_rows:
        c          = np.zeros(n_coeffs, dtype=np.float64)
        c[:len(row)] = np.asarray(row, dtype=np.float64)
        coefficients.append(c)

    return {
        "granule_edges": granule_edges, 
        "coefficients": coefficients,
        "order": n_coeffs - 1
    }
