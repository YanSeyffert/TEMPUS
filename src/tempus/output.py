# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/output.py
# handles CSV output and .run.json metadata for TEMPUS ephemeris time series. 
# Chebyshev outputs are handled in the cli.py via the src/tempus/chebycode sub-module

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import json
import re
from typing import Dict, Optional

from numpy.polynomial import chebyshev as C
from .model import jd_parts_to_et

# ---- Time-scale constants (for TT<->TCG scaling) ----
# From IAU Definitions
LG = 6.969290134e-10
T0_JD = 2443144.5003725
DAY_S = 86400.0
JD_J2000 = 2451545.0

# ---- Main output routines ----

def short_body_name(key: str) -> str:
    """Return a short label for a body key for use in column/timescale names.

    Standard IAU timescale letters are used where defined:
      EARTH_CENTER -> G  (TCG)
      MOON         -> L  (TCL)
    For other bodies, trailing _CENTER or _BARYCENTER is stripped.
    """
    if key == "EARTH_CENTER":
        return "G"
    if key == "MOON":
        return "L"
    for suffix in ("_CENTER", "_BARYCENTER"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def auto_out_name(target_ref, center_ref, grid, eph_path, asteroid_file=None) -> str:
    """
    Automatic output filename: {tcx}_{mYYY}_{pYYY}_{ephem-stem}[_ast].csv with
    the timescale tag from the target (tcg, tcl, tcmars, ...) and the grid span
    as whole years before (m) / after (p) J2000. A non-SSB center is prefixed
    (e.g. tcg_tcl_...). An `_ast` tag is appended when a perturbing asteroid
    file was supplied.
    """
    def ytag(jd_int, jd_frac, *, is_start):
        y = (int(jd_int) - 2451545 + float(jd_frac)) / 365.25
        rounded = round(abs(y))
        # Tie-break a boundary that *displays* as 000 by position, not by the
        # sign of y. Grids rarely start at bit-exact J2000: e.g. TT->TCB
        # conversion shifts a nominal J2000 start by the ~11s TCB-TT drift
        # since the 1977 origin, leaving a tiny positive y that rounds to 000
        # but fails "y < 0" — that must still read "m", not "p".
        is_before = y < 0 if rounded else is_start
        return ("m" if is_before else "p") + f"{rounded:03d}"

    label = "tc" + short_body_name(target_ref.key).lower()
    if center_ref.naif != 0:
        label = "tc" + short_body_name(center_ref.key).lower() + "_" + label

    # INPOP files ship with a distribution suffix encoding the file's own span
    # and argument-timescale tag, e.g. inpop19a_TCB_m100_p100_tcg. Drop that
    # trailing _m<yr>_p<yr>_<ts>: our own _mYYY_pYYY_ already carries the
    # grid span, so keeping it would duplicate the span and tack on a redundant
    # timescale tag. The ephemeris flavour (…_tcb / …_tdb) is kept.
    eph_stem = re.sub(r"_m\d+_p\d+_[a-z]+$", "", Path(eph_path).stem.lower())

    name = (f"{label}_{ytag(grid.jd_int[0], grid.jd_frac[0], is_start=True)}"
            f"_{ytag(grid.jd_int[-1], grid.jd_frac[-1], is_start=False)}"
            f"_{eph_stem}"
            f"{'_ast' if asteroid_file else ''}"
            ".csv")

    # Prefer the project's output area over CWD: ./output for runs launched
    # from the project root, ../output for runs from a working subfolder
    # (examples/, notebooks/).
    for out_dir in (Path("output"), Path("..") / "output"):
        if out_dir.is_dir():
            return str(out_dir / name)
    return name


@dataclass(frozen=True)
class OutputRequest:
    """
    Which columns to output.

    Always output:
      - JD_int, JD_frac (split Julian Date of ephemeris argument timescale)
      - SecondsPastJ2000 (seconds past J2000 in ephemeris timescale)
      - TCB_minus_TCX (seconds)

    Optionals:
      - TCG_minus_TCB (queried, if available)
      - TT_minus_TDB (queried, if available)
      - TT_minus_TCG (computed if we can form JD_TT)
      - dTCX_dTCB (instantaneous rate)
    """
    include_tcg_minus_tcb: bool = False
    include_tt_minus_tdb: bool = False
    include_tt_minus_tcg: bool = False
    include_dtcx_dtcb: bool = False


def tt_minus_tcg_from_jdtt(jd_tt: np.ndarray) -> np.ndarray:
    """
    Compute TT - TCG from JD_TT using IERS Eq. (10.1)
    Returns seconds.

    TT - TCG = - (LG/(1-LG)) * (JD_TT - T0) * 86400
    """
    return -(LG / (1.0 - LG)) * (jd_tt - T0_JD) * DAY_S


def write_csv(path: str | Path, cols: Dict[str, np.ndarray], meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure consistent order for time columns.
    keys = list(cols.keys())
    preferred = ["JD_int", "JD_frac", "SecondsPastJ2000"]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

    arrays = [np.asarray(cols[k]) for k in keys]
    data = np.column_stack(arrays)
    header = ",".join(keys)
    fmt = ["%.0f" if k == "JD_int" else "%.17e" for k in keys]

    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmt)
    if meta is not None:
        write_run_json(path.with_suffix(".run.json"), meta)


def read_csv(path: str | Path) -> tuple[Dict[str, np.ndarray], Optional[dict]]:
    """
    Read a CSV written by write_csv. Returns (cols, meta), where meta is the
    content of the sibling .run.json (None if absent). JD_int comes back as
    int64, everything else as float64.
    """
    path = Path(path)
    with path.open() as f:
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: header has {len(names)} columns, data has {data.shape[1]}.")
    cols = {n: (data[:, i].astype(np.int64) if n == "JD_int" else data[:, i])
            for i, n in enumerate(names)}
    meta_path = path.with_suffix(".run.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    return cols, meta


def write_run_json(path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps_pretty_json(meta) + "\n")

# Makes lists of specified keys, on one line, instead of line breaks after each list item.
def _dumps_pretty_json(meta: dict) -> str:
    compact_list_keys = {
        "asteroid_indices", 
        #"ext_resolved",
    }

    def fmt(value, level: int, parent_key: str | None = None) -> str:
        pad = " " * (2 * level)
        child_pad = " " * (2 * (level + 1))

        if isinstance(value, dict):
            if not value:
                return "{}"
            items = list(value.items())
            lines = []
            for i, (k, v) in enumerate(items):
                rendered = fmt(v, level + 1, k)
                line = f"{child_pad}{json.dumps(k)}: {rendered}"
                if i < len(items) - 1:
                    line += ","
                lines.append(line)
            return "{\n" + "\n".join(lines) + f"\n{pad}" + "}"

        if isinstance(value, list):
            if parent_key in compact_list_keys:
                return json.dumps(value, separators=(", ", ": "))
            if not value:
                return "[]"
            lines = []
            for i, item in enumerate(value):
                rendered = fmt(item, level + 1, None)
                line = f"{child_pad}{rendered}"
                if i < len(value) - 1:
                    line += ","
                lines.append(line)
            return "[\n" + "\n".join(lines) + f"\n{pad}" + "]"

        return json.dumps(value)

    return fmt(meta, 0)


def build_columns(
    *,
    tcb_minus_tcx: np.ndarray,
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    eph=None,
    req: OutputRequest = OutputRequest(),
    dtcx_dtcb: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Build output columns. `eph` is optional, but required for time-ephemeris queries.

    Required inputs:
      - tcb_minus_tcx: seconds
      - jd_int, jd_frac: split Julian Date arrays

    Optional:
      - eph.time_offset_jd(kind, jd_int, jd_frac)
        (CALCEPH): provides TCG-TCB and/or TT-TDB
    """
    ji = np.asarray(jd_int, dtype=np.int64).reshape(-1)
    jf = np.asarray(jd_frac, dtype=np.float64).reshape(-1)
    if ji.size != jf.size:
        raise ValueError("jd_int and jd_frac must have the same size.")
    et_arr = jd_parts_to_et(ji, jf)

    cols: Dict[str, np.ndarray] = {
        "JD_int": ji,
        "JD_frac": jf,
        "SecondsPastJ2000": et_arr,
        "TCB_minus_TCX": np.asarray(tcb_minus_tcx, dtype=np.float64),
    }

    # --- Instantaneous rate dTCX/dTCB ---
    if req.include_dtcx_dtcb:
        if dtcx_dtcb is None:
            raise RuntimeError("Requested dTCX_dTCB but no rate array was provided to build_columns().")
        cols["dTCX_dTCB"] = np.asarray(dtcx_dtcb, dtype=np.float64)

    # --- Queried columns (from ephemeris time-ephemeris if supported, like INPOP) ---
    if req.include_tcg_minus_tcb:
        if eph is None or not hasattr(eph, "time_offset_jd"):
            raise RuntimeError("Requested TCG_minus_TCB but ephemeris backend has no time_offset_jd().")
        vals = eph.time_offset_jd("TCG-TCB", ji, jf)
        cols["TCG_minus_TCB"] = np.asarray(vals, dtype=np.float64)

    if req.include_tt_minus_tdb:
        if eph is None or not hasattr(eph, "time_offset_jd"):
            raise RuntimeError("Requested TT_minus_TDB but ephemeris backend has no time_offset_jd().")
        vals = eph.time_offset_jd("TT-TDB", ji, jf)
        cols["TT_minus_TDB"] = np.asarray(vals, dtype=np.float64)

    # --- Computed TT_minus_TCG if requested and feasible ---
    if req.include_tt_minus_tcg:
        # Minimal safe path: only do this if you can form JD_TT.
        # If ephemeris timescale is TDB and you have TT-TDB, you can construct JD_TT:
        if "TT_minus_TDB" not in cols:
            raise RuntimeError("Requested TT_minus_TCG but TT_minus_TDB not available to construct JD_TT.")
        # JD_TDB = JD_J2000 + et/86400.0, then JD_TT = JD_TDB + (TT-TDB)/86400.
        # Note: This assumes the ephemeris argument is TDB-like.
        jd_tdb = cols["JD_int"].astype(np.float64) + cols["JD_frac"]
        jd_tt = jd_tdb + cols["TT_minus_TDB"] / DAY_S
        cols["TT_minus_TCG"] = tt_minus_tcg_from_jdtt(jd_tt)

    return cols
