# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/cli.py
# this file defines the command-line interface for TEMPUS

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

from . import __version__
from .model import Delta_TCX, Delta_TCX_TCY, jd_grid, jd_tdb_to_tcb, per_leg_ext_bodies, DAY_S, L_B
from .bodies import parse_body, ext_list, DEFAULT_EXT_KEYS, NAIF


def _path_for_metadata(path: str | Path) -> str:
    """
    Use a TEMPUS-root-relative path when ephemeris is under data/, else absolute.
    """
    p = Path(path).expanduser().resolve()
    tempus_root = Path(__file__).resolve().parents[2]
    data_root = (tempus_root / "data").resolve()
    try:
        p.relative_to(data_root)
        return p.relative_to(tempus_root).as_posix()
    except ValueError:
        return str(p)


def _parse_int_csv(raw: str | None) -> list[int]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_ext_arg(ext_arg: str | None, target_ref):
    """
    ext_arg:
      - None or "auto"  -> derive default ext list excluding target
      - comma-separated list -> parse each element as a body (id or name)
    Returns list[BodyRef]
    """
    if ext_arg is None or ext_arg.strip().lower() == "auto":
        return ext_list(target_ref)

    parts = [p.strip() for p in ext_arg.split(",") if p.strip()]
    if not parts:
        return ext_list(target_ref)

    refs = [parse_body(p) for p in parts]
    # Ensure target excluded even if user accidentally included it
    return [r for r in refs if r.naif != target_ref.naif]


def _cmd_integrate(args: argparse.Namespace) -> int:
    from .ephem import CalcephEphem
    from .output import OutputRequest, auto_out_name, build_columns, short_body_name, write_csv

    target_ref = parse_body(args.target)
    center_ref = parse_body(args.center)
    # Non-SSB center -> BCRS-subtraction difference TC_center - TC_target
    # ("native subtraction", model.Delta_TCX_TCY).
    diff_mode = center_ref.naif != 0
    if diff_mode and target_ref.naif == center_ref.naif:
        raise ValueError("--target and --center must be different bodies.")

    if diff_mode and (args.ext is None or args.ext.strip().lower() == "auto"):
        # Union ext list for both legs (Delta_TCX_TCY excludes per leg). If
        # Earth or Moon is involved, split EMB into the two point masses so no
        # leg ends up with both EMB and a component (double-counted system).
        base = list(DEFAULT_EXT_KEYS)
        if {target_ref.naif, center_ref.naif} & {NAIF["EARTH_CENTER"], NAIF["MOON"]}:
            i = base.index("EARTH_MOON_BARYCENTER")
            base[i:i + 1] = ["EARTH_CENTER", "MOON"]
        ext_refs = [r for r in (parse_body(k) for k in base)
                    if r.naif not in (target_ref.naif, center_ref.naif)]
    else:
        ext_refs = _parse_ext_arg(args.ext, target_ref)

    if args.integrator == "amk" and args.amk_k is None:
        raise ValueError("Integrator 'amk' requires --amk-k.")

    # Check every input file up front: a missing download then gets a clear
    # message instead of a CALCEPH error that may name a different file.
    inputs = [("--eph-file", p) for p in args.eph_file]
    inputs += [("--asteroid-file", args.asteroid_file)] if args.asteroid_file else []
    inputs += [("--asteroid-gm-header", p) for p in (getattr(args, "asteroid_gm_header", None) or [])]
    for flag, raw in inputs:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{flag}: file not found: {path.resolve()} "
                                    "(public ephemerides: scripts/download_data.sh)")

    eph = CalcephEphem.open(args.eph_file, asteroid_file=args.asteroid_file)
    try:
        # Print ephemeris version if available and timescale info
        ver = eph.fileversion() if hasattr(eph, "fileversion") else None
        if ver:
            print(f"Ephemeris version: {ver}")

        ts_enum = eph.timescale_enum()
        ts_label = eph.timescale()
        if ts_enum is not None:
            print(f"CALCEPH timescale: {ts_enum} ({ts_label})")

        # Supplement GM constants from an external DE header file before the
        # asteroid GM availability check.  This allows using e.g. a DE441 header
        # alongside a DE440 main ephemeris when the asteroid BSP was built for
        # a different release.
        for gm_header_raw in (getattr(args, "asteroid_gm_header", None) or []):
            gm_header_path = Path(gm_header_raw).expanduser().resolve()
            if not gm_header_path.is_file():
                raise FileNotFoundError(f"--asteroid-gm-header: file not found: {gm_header_path}")
            added, overridden_keys = eph.supplement_gm_from_header(gm_header_path)
            parts = []
            if added:
                parts.append(f"{added} new")
            if overridden_keys:
                parts.append(f"{len(overridden_keys)} overridden ({', '.join(overridden_keys)})")
            summary = ", ".join(parts) if parts else "no changes"
            print(f"Loaded GM header '{gm_header_path.name}': {summary} (header values take priority).")

        # Asteroid handling: --asteroid-ids takes either asteroid indices
        # (< 2000000, e.g. 90482 for Orcus) or NAIF SPK IDs (>= 2000000,
        # e.g. 2090482); both normalise to indices.
        requested_asteroid_indices = list(dict.fromkeys(
            CalcephEphem.asteroid_index_from_any(v)
            for v in _parse_int_csv(getattr(args, "asteroid_ids", None))
        ))
        if requested_asteroid_indices and not args.asteroid_file:
            raise ValueError("--asteroid-ids requires --asteroid-file.")

        asteroid_mode = "none"
        asteroid_indices: list[int] = []
        skipped_not_in_file_indices: list[int] = []
        skipped_no_gm_indices: list[int] = []
        asteroid_log_line = None

        if args.asteroid_file:
            file_obj_ids = eph.asteroid_object_ids_in_file()
            file_idx_set = set(CalcephEphem.asteroid_indices_from_object_ids(file_obj_ids))
            gm_idx_set = set(eph.asteroid_indices_with_gm_in_main_file())
            asteroid_mode = "explicit" if requested_asteroid_indices else "auto"
            if asteroid_mode == "explicit":
                skipped_not_in_file_indices = [i for i in requested_asteroid_indices if i not in file_idx_set]
                skipped_no_gm_indices = [i for i in requested_asteroid_indices if i in file_idx_set and i not in gm_idx_set]
                source_indices = requested_asteroid_indices
            else:
                source_indices = sorted(file_idx_set)
                skipped_no_gm_indices = sorted(file_idx_set - gm_idx_set)
            asteroid_indices = [i for i in source_indices if i in file_idx_set and i in gm_idx_set]

        added_asteroid_count = 0
        if asteroid_indices:
            base_ids = {b.naif for b in ext_refs}
            for oid in CalcephEphem.asteroid_object_ids(asteroid_indices):
                if oid != target_ref.naif and oid not in base_ids:
                    ext_refs.append(parse_body(oid))
                    base_ids.add(oid)
                    added_asteroid_count += 1
            src = "auto-discovered" if asteroid_mode == "auto" else "requested"
            asteroid_log_line = f"Using {added_asteroid_count} {src} asteroids (positions + GM available)."

        if skipped_not_in_file_indices:
            print(
                f"Skipped {len(skipped_not_in_file_indices)} requested asteroids not present in '{args.asteroid_file}'."
            )
        if skipped_no_gm_indices:
            gm_hint = (
                ""
                if getattr(args, "asteroid_gm_header", None)
                else " Use --asteroid-gm-header with a DE header style file to load asteroid GMs"
                     " (e.g. header.441 from https://ssd.jpl.nasa.gov/ftp/eph/planets/ascii/de441/)."
            )
            print(
                f"Skipped {len(skipped_no_gm_indices)} asteroids without GM constants.{gm_hint}"
            )
        if asteroid_log_line:
            print(asteroid_log_line)

        # Build split-JD time grid; ET is derived downstream when needed.
        # JD inputs are in the ephemeris's native timescale. With
        # --input-jd-are-tdb on a TCB-scale ephemeris, start/end/step are all
        # TDB: the grid is built in TDB and converted point-by-point to TCB via
        # the exact IAU 2006 Res. B3 linear relation, so each epoch is derived
        # independently and no rounding accumulates along the grid.
        jd_step = np.float64(args.jd_step)
        input_jd_are_tdb = bool(getattr(args, "input_jd_are_tdb", False))
        convert_tdb_grid = False
        if input_jd_are_tdb:
            if ts_enum == 2:  # TCB
                convert_tdb_grid = True
            elif ts_enum != 1:
                raise ValueError(f"--input-jd-are-tdb requires TDB or TCB ephemeris, got timescale_enum={ts_enum}.")

        grid = jd_grid(
            jd_start_int=int(args.jd_start[0]), jd_start_frac=np.float64(args.jd_start[1]),
            jd_end_int=int(args.jd_end[0]),     jd_end_frac=np.float64(args.jd_end[1]),
            jd_step=jd_step,
        )
        jd_step_native = jd_step
        if convert_tdb_grid:
            l_b = np.float64(eph.constants().get("LBTCBD", L_B))
            grid = jd_tdb_to_tcb(grid, l_b=l_b)
            jd_step_native = jd_step / (np.float64(1.0) - l_b)
            print(f"  Converted TDB input grid to TCB (IAU 2006 B3, L_B={l_b:.12e}); native TCB step: {float(jd_step_native):.17g} d")

        # GM provider directly from ephemeris constants, converted to SI
        gm_provider = lambda b: eph.gm_m3_s2(int(b))

        include_sun_j2 = bool(getattr(args, "sun_j2", False))
        if include_sun_j2:
            j2_sun, r_sun_m = eph.sun_j2_constants()
            print(f"Sun J2: J2SUN={j2_sun:.6e}, R_Sun={r_sun_m:.6e} m")

        model_kwargs = dict(
            eph=eph,
            jd_int=grid.jd_int,
            jd_frac=grid.jd_frac,
            target=target_ref.naif,
            ext_bodies=[b.naif for b in ext_refs],
            gm_provider=gm_provider,
            integrator=args.integrator,
            amk_k=args.amk_k,
            include_o4=not args.pn1,
            use_longdouble=args.longdouble,
            include_sun_j2=include_sun_j2,
        )
        if diff_mode:
            delta, rate = Delta_TCX_TCY(center=center_ref.naif, **model_kwargs)
            c_short = short_body_name(center_ref.key)
            t_short = short_body_name(target_ref.key)
            drift_label = f"TC{c_short}-TC{t_short}"
            col_name_main = f"TC{c_short}_minus_TC{t_short}"
            # Mirrors the 4th line in Delta_TCX_TCY:
            ext_leg_target, ext_leg_center = per_leg_ext_bodies(target_ref.naif, center_ref.naif, [b.naif for b in ext_refs])
        else:
            delta, rate = Delta_TCX(center=center_ref.naif, **model_kwargs)
            drift_label = "TCB-TCX"
            col_name_main = "TCB_minus_TCX"

        span_days = float((grid.jd_int[-1] - grid.jd_int[0]) + (grid.jd_frac[-1] - grid.jd_frac[0]))
        drift_ms_per_day = (delta[-1] - delta[0]) / span_days * 1e3 if span_days > 0.0 else np.nan
        print(f"Approx. mean drift ({drift_label}): {drift_ms_per_day:.6f} ms/day")

        # Optional extra columns via ephemeris time offsets and derived combos
        req = OutputRequest(
            include_tcg_minus_tcb=getattr(args, "out_tcg_tcb", False),
            include_tt_minus_tdb=getattr(args, "out_tt_tdb", False),
            include_tt_minus_tcg=getattr(args, "out_tt_tcg", False),
            include_dtcx_dtcb=getattr(args, "out_rate", False),
        )

        cols = build_columns(
            jd_int=grid.jd_int,
            jd_frac=grid.jd_frac,
            tcb_minus_tcx=delta,
            eph=eph,
            req=req,
            dtcx_dtcb=rate,
        )

        if diff_mode:
            rename = {"TCB_minus_TCX": col_name_main, "dTCX_dTCB": f"d{col_name_main}_dTCB"}
            cols = {rename.get(k, k): v for k, v in cols.items()}

        meta = {
            "tempus_version": __version__,
            "backend": "calceph",
            "ephem_file": _path_for_metadata(eph.path),
            "ephem_version": ver,
            "time_columns": ["JD_int", "JD_frac", "SecondsPastJ2000"],
            "in_timescale": ts_label,
            "calceph_timescale_enum": ts_enum,
            "target": args.target,
            "center": args.center,
            "non_ssb_center_mode": diff_mode,
            **({"difference_mode": "bcrs_subtraction", "simultaneity": "TCB"} if diff_mode else {}),
            "ext_arg": args.ext,
            "asteroid_file": _path_for_metadata(eph.asteroid_path) if eph.asteroid_path else None,
            "asteroid_gm_header": [_path_for_metadata(p) for p in args.asteroid_gm_header] if getattr(args, "asteroid_gm_header", None) else None,
            "asteroid_mode": asteroid_mode,
            "asteroid_indices": asteroid_indices,
            **({"ext_resolved": [b.key for b in ext_refs]} if not diff_mode else {}),
            **({"ext_resolved_target": [parse_body(n).key for n in ext_leg_target],
                "ext_resolved_center": [parse_body(n).key for n in ext_leg_center]}
               if diff_mode else {}),
            "jd_start": args.jd_start[0],
            "jd_start_frac": args.jd_start[1],
            "jd_end": args.jd_end[0],
            "jd_end_frac": args.jd_end[1],
            "jd_step_input_days": args.jd_step,
            "input_jd_are_tdb": input_jd_are_tdb,
            # Native-timescale (converted) grid values, only when a TDB->TCB conversion actually happened.
            **({"input_timescale": "TDB",
                "jd_step_native_days": float(jd_step_native),
                "jd_grid_native_start": [int(grid.jd_int[0]), float(grid.jd_frac[0])],
                "jd_grid_native_end": [int(grid.jd_int[-1]), float(grid.jd_frac[-1])]}
               if convert_tdb_grid else {}),
            "duration_days": span_days,
            "integrator": args.integrator,
            "amk_k": args.amk_k,
            "include_pn2": not bool(args.pn1),
            "use_longdouble": bool(args.longdouble),
            "include_sun_j2": bool(getattr(args, "sun_j2", False)),
            "columns": list(cols.keys()),
            # The column this run's integration actually produced (as opposed
            # to a definitional/ephemeris-native extra column). Its zero is a
            # free constant (the only column `chebyshev --anchor` may act on)
            # and it's the default `chebyshev --column` target.
            "tempus_integrated_column": col_name_main,
        }

        out_path = args.out or auto_out_name(target_ref, center_ref, grid, eph.path,
                                              asteroid_file=args.asteroid_file)
        write_csv(Path(out_path), cols, meta=meta)
        print(f"Wrote {out_path}")
        return 0

    finally:
        eph.close()



def _cmd_chebyshev(args: argparse.Namespace) -> int:
    from .chebycode import ChebyshevSeries
    from .chebycode.validation import (
        compare_against_samples,
        print_report,
        truncation_error_estimate,
    )
    from .output import read_csv

    if args.dist == 1 and not args.use_rate:
        raise ValueError("--dist 1 (C1 boundary conditions) requires --use-rate.")

    cols, meta = read_csv(args.csv)
    time_cols = set((meta or {}).get("time_columns", ["JD_int", "JD_frac", "SecondsPastJ2000"]))

    # Default target is the run's own record of what it integrated (accurate
    # for both TCB_minus_TCX and diff_mode's dynamic names)
    column = args.column or (meta or {}).get("tempus_integrated_column", "TCB_minus_TCX")
    if column not in cols:
        fittable = sorted(k for k in cols if k not in time_cols)
        raise ValueError(f"Column '{column}' not found in {args.csv}. Available: {fittable}")

    jd_int, jd_frac, y = cols["JD_int"], cols["JD_frac"], cols[column]

    dy = None
    if args.use_rate:
        rate_col = "dTCX_dTCB" if column == "TCB_minus_TCX" else f"d{column}_dTCB"
        if rate_col not in cols:
            raise ValueError(f"--use-rate: rate column '{rate_col}' not found; run integrate with --out-rate.")
        # y = TCB - TCX [s] against JD in the grid timescale; dTCB per grid day
        # differs from 86400 s only by O(L_B) — negligible against fit residuals.
        dy = (np.float64(1.0) - cols[rate_col]) * DAY_S

    # Anchoring (opt-in): zero the series at a chosen epoch; bare --anchor
    # means the 1977 origin, where all TCX read the same JD by definition.
    # A standalone value convention; but note that --also-wrt-tcx builds the TCX
    # axis from Delta itself (T = t - Delta/86400), so the value's zero
    # convention also fixes the zero convention of the TCX argument labels.
    anchor = None
    if args.anchor is not None:
        # Only the column TEMPUS's own integration left as a free constant may
        # be re-zeroed. Columns with a definitional zero (TT_minus_TDB,
        # TCG_minus_TCB, TT_minus_TCG, ...) are excluded by not being it;
        # anchoring them would silently redefine a T0.
        integrated = (meta or {}).get("tempus_integrated_column", "TCB_minus_TCX")
        if column != integrated:
            raise ValueError(
                f"--anchor only applies to '{integrated}' (the column this run's "
                f"integration left as a free constant); '{column}' has a zero "
                "fixed by definition, not by TEMPUS's integration."
            )
        if len(args.anchor) == 0:
            anchor = (2443144, np.float64(0.5003725))  # 1977-01-01.0 TAI origin
        elif len(args.anchor) == 2:
            anchor = (int(args.anchor[0]), np.float64(args.anchor[1]))
        else:
            raise ValueError("--anchor takes either no values (1977 origin) or two: INT FRAC.")
        lo = (int(jd_int[0]), float(jd_frac[0]))
        hi = (int(jd_int[-1]), float(jd_frac[-1]))
        if not (lo <= (anchor[0], float(anchor[1])) <= hi):
            raise ValueError(f"--anchor epoch {anchor[0]}+{anchor[1]} lies outside the CSV span {lo}..{hi}.")

    def _fit(ji, jf, yy, dyy, order: int, granule_len: float, dist: int):
        cs = ChebyshevSeries.from_samples(
            ji, jf, yy, dyy,
            granule_len=granule_len, order=order, npoints=args.npoints, dist=dist,
        )
        stats = compare_against_samples(cs, ji, jf, yy)
        conv = truncation_error_estimate(cs)
        # With an explicit error budget, the sample residual is the binding
        # gate: near machine-precision fits the coefficient tail flattens into
        # the float64 noise floor and `converged` reads False even though the
        # fit is excellent. Without a budget, tail convergence is all we have.
        if args.max_err is not None:
            ok = stats["max_abs_error"] <= args.max_err
        else:
            ok = bool(conv["converged"])
        return cs, stats, ok

    validate = not args.no_validate
    order_used, granule_used = args.order, args.granule_len
    cs, stats, ok = _fit(jd_int, jd_frac, y, dy, order_used, granule_used, args.dist)
    if validate and not ok and args.auto:
        for granule_len in (args.granule_len, args.granule_len / 2.0, args.granule_len / 4.0):
            for order in range(args.order, 16):
                if (order, granule_len) == (args.order, args.granule_len):
                    continue  # already fitted above
                cs, stats, ok = _fit(jd_int, jd_frac, y, dy, order, granule_len, args.dist)
                # Record the parameters of whatever `cs` now holds, not just of a
                # successful fit: when every combination fails, `cs` is the last
                # attempt, and the --also-wrt-tcx companion must be fitted the
                # same way or the pair on disk would disagree.
                order_used, granule_used = order, granule_len
                print(f"  --auto: order={order} granule_len={granule_len:g} -> "
                      f"max_err={stats['max_abs_error']:.3e} ok={ok}")
                if ok:
                    break
            if ok:
                break

    if anchor is not None:
        # A constant shift moves only each granule's c0 coefficient (T_0 = 1);
        # all other coefficients are unchanged (verified: <1e-12 vs. a full
        # refit on shifted samples). Patch coefficients directly instead of
        # re-fitting the whole series a second time on shifted samples.
        shift = float(cs.evaluate(np.array([anchor[0]], dtype=np.int64), np.array([anchor[1]]))[0])
        for gran in cs.granules:
            gran.coeffs[0] -= shift
        y = y - shift          # samples too: they carry the TCX labels below
        stats = {**stats, "mean_error": stats["mean_error"] - shift}
        print(f"  anchored: series zeroed at JD {anchor[0]} + {float(anchor[1])} (shift {-shift:+.9f} s)")

    # Foolproofing: every .asc carries which clock labels its axis and
    # where the series is zeroed, so two files can never be silently
    # subtracted across a mismatched argument timescale or zero convention
    # (see notebooks/09_argument_timescale.ipynb).
    native_ts = (meta or {}).get("t_timescale") or "native"
    zero_tag = (f"zero=JD{anchor[0]}+{float(anchor[1]):.7f}" if anchor is not None
                else "zero=integration-start")
    title_bits = ""
    if meta:
        bits = [meta.get("target"), Path(meta.get("ephem_file") or "").stem or None]
        title_bits = " — " + " ".join(str(b) for b in bits if b)

    print_report(cs)
    print(f"  Samples      max  error : {stats['max_abs_error']:.5e}")
    print(f"               RMS  error : {stats['rms_error']:.5e}")

    native_path = Path(args.out) if args.out else Path(args.csv).with_suffix(".asc")
    cs.to_asc(str(native_path),
             title=f"ChebyshevSeries — {column}{title_bits} — argument_timescale={native_ts} {zero_tag}")
    print(f"Wrote {native_path} (wrt {native_ts})")

    ok_all = ok
    if args.also_wrt_tcx:
        from .model import TimeGrid, relabel_to_tcx
        # Exact per-sample relabelling; the values y are unchanged. TCX is
        # never written alone: it always accompanies the native product above,
        # sharing the same run, fit settings and anchor.
        tcx_grid = relabel_to_tcx(TimeGrid(jd_int, jd_frac), y)
        # No derivative: the rate column is dy/dt of the native argument, not dy/dT.
        cs_tcx, stats_tcx, ok_tcx = _fit(tcx_grid.jd_int, tcx_grid.jd_frac, y, None, order_used, granule_used, 2)
        ok_all = ok_all and ok_tcx

        print()
        print_report(cs_tcx)
        print(f"  Samples      max  error : {stats_tcx['max_abs_error']:.5e} (wrt TCX)")
        print(f"               RMS  error : {stats_tcx['rms_error']:.5e}")

        tcx_path = Path(str(Path(args.csv).with_suffix("")) + "_tcx.asc")
        cs_tcx.to_asc(str(tcx_path),
                      title=f"ChebyshevSeries — {column}{title_bits} — argument_timescale=TCX {zero_tag}")
        print(f"Wrote {tcx_path} (wrt TCX)")

    if validate and not ok_all:
        hint = (
            " Note: near machine-precision fits report converged=False; set --max-err "
            "to gate on your actual residual budget instead."
            if args.max_err is None else ""
        )
        print(
            f"FIT-QUALITY GATE FAILED (see report above); use --auto, raise --order, or shrink --granule-len.{hint}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tempus",
        description="TEMPUS: ephemeris-driven time-coordinate integration",
    )
    p.add_argument("--version", action="version", version=f"tempus {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # CALCEPH
    # -------------------------
    pc = sub.add_parser("integrate", help="Run timescale integration using CALCEPH backend")
    pc.add_argument("--eph-file", required=True, nargs="+", help="Path(s) to ephemeris files (.bsp, .tls, .tpc, etc.)")

    pc.add_argument("--jd-start", nargs=2, type=float, metavar=("INT", "FRAC"),
                    required=True, help="Start JD as integer and fractional parts, e.g. --jd-start 2451545 0.000130251")
    pc.add_argument("--jd-end",   nargs=2, type=float, metavar=("INT", "FRAC"),
                    required=True, help="End JD as integer and fractional parts, e.g. --jd-end 2451555 0.0")
    pc.add_argument("--jd-step",  type=float, required=True, help="Step size in days.")

    # Backend-agnostic body inputs (id or name)
    pc.add_argument("--target", default="MARS_BARYCENTER", help='Target body (e.g. "MARS_BARYCENTER" or "4")')
    pc.add_argument(
        "--center", default="SSB",
        help=(
            'Center body (e.g. "SSB" or "0"). A non-SSB center (e.g. "EARTH_CENTER" '
            "with --target MOON) computes the difference TC_center - TC_target as a "
            "function of TCB via BCRS subtraction of two SSB-centered integrations."
        ),
    )

    # External bodies: auto or comma-separated
    pc.add_argument("--ext", default="auto",
        help='External bodies for U_ext: "auto" or comma-separated list (e.g. "SUN,EARTH_BARYCENTER,5")',
    )

    # External bodies: asteroids
    pc.add_argument(
        "--asteroid-file",
        default=None,
        help="Optional path to auxiliary asteroid ephemeris file used for asteroid object states.",
    )
    pc.add_argument(
        "--asteroid-ids",
        default=None,
        help="Optional comma-separated asteroid indices. Requires --asteroid-file. Only asteroids present in --asteroid-file and with GM constants are included.",
    )
    pc.add_argument(
        "--asteroid-gm-header",
        default=None,
        nargs="+",
        metavar="PATH",
        help=(
            "One or more JPL ASCII DE header files (e.g. header.441 header_kbo.441) from which to "
            "supplement missing asteroid GM constants. Useful when the asteroid BSP "
            "was generated for a different DE release than the main ephemeris "
            "(e.g. sb441-n373s.bsp with de440.bsp). Constants from these files are "
            "added only if not already present in the main ephemeris."
        ),
    )

    # Integration method
    pc.add_argument(
        "--integrator",
        default="am12",
        help="Integrator name (default: am12). Options: trapz, rk4, am4, am8, am10, am12, amk.",
    )
    pc.add_argument(
        "--amk-k",
        type=int,
        default=None,
        help="k-step for amk (required when --integrator=amk). Example: 11 for AM12.",
    )
    pc.add_argument(
        "--sun-j2", action="store_true",
        help=(
            "Include the Sun's J2 (quadrupole) contribution to the gravitational potential "
            "(DeltaU = -GM*J2*R^2/r^3 * P2(sin phi), phi = heliographic latitude of target). "
            "J2SUN and RSUN/ASUN are read from the ephemeris header constants."
        ),
    )
    pc.add_argument(
        "--pn1", action="store_true",
        help="Use only 1PN (O(c^-2)) accuracy; omit the 2PN (O(c^-4)) correction (default: 2PN included).",
    )
    pc.add_argument(
        "--longdouble", action="store_true",
        help="Use longdouble for model computations (experimental; ephemeris input remains float64).",
    )
    # Input time scale
    pc.add_argument(
        "--input-jd-are-tdb", action="store_true",
        help=(
            "By default all JD inputs (--jd-start/--jd-end/--jd-step) are in the "
            "ephemeris's native timescale (TDB for most ephemerides; TCB for the "
            "TCB releases of INPOP). With this flag the inputs are TDB even for a "
            "TCB-scale ephemeris: the grid is built in TDB and every point is "
            "converted to TCB exactly (IAU 2006 Res. B3, linear). "
            "Has no effect on TDB-scale ephemerides."
        ),
    )

    # Output filename
    pc.add_argument(
        "--out", default=None,
        help=(
            "Output CSV path. Default: auto-generated as "
            "{tcx}_{mYYYY}_{pYYYY}_{ephem}.csv from the target timescale, the "
            "grid span in whole years before/after J2000, and the ephemeris name."
        ),
    )

    # Optional extra output columns
    pc.add_argument(
        "--out-tcg-tcb", action="store_true",
        help="Include TCG_minus_TCB column if the ephemeris supports it (CALCEPH time ephemeris).",
    )
    pc.add_argument(
        "--out-tt-tdb", action="store_true",
        help="Include TT_minus_TDB column if the ephemeris supports it (CALCEPH time ephemeris).",
    )
    pc.add_argument(
        "--out-tt-tcg", action="store_true",
        help="Include TT_minus_TCG column (computed via IERS/IAU linear scaling; currently requires --out-tt-tdb).",
    )
    pc.add_argument(
        "--out-rate", action="store_true",
        help="Include dTCX_dTCB column: instantaneous dTCX/dTCB rate at each grid point.",
    )

    # Set command handler
    pc.set_defaults(func=_cmd_integrate)

    # -------------------------
    # Chebyshev fitting
    # -------------------------
    ph = sub.add_parser(
        "chebyshev",
        help="Fit a piecewise Chebyshev series to an integrate CSV and write a .asc file",
    )
    ph.add_argument("csv", help="CSV file produced by 'tempus integrate'")
    ph.add_argument("--column", default=None,
                    help="Column to fit (default: the run's tempus_integrated_column from "
                         "metadata, i.e. TCB_minus_TCX for a standard run).")
    ph.add_argument("--order", type=int, default=11,
                    help="Polynomial degree per granule (default: 11).")
    ph.add_argument("--granule-len", type=float, default=4.0,
                    help="Granule window length in days (default: 4).")
    ph.add_argument("--npoints", type=int, default=10,
                    help="Stencil size for local interpolation (default: 10).")
    ph.add_argument("--dist", type=int, default=2, choices=(1, 2, 3),
                    help="Node distribution (default 2: Gauss-Lobatto with endpoints; 1 requires --use-rate).")
    ph.add_argument("--use-rate", action="store_true",
                    help="Use the rate column (integrate --out-rate) as derivative information in the fit.")
    ph.add_argument("--max-err", type=float, default=None,
                    help="Fit-quality gate: fail if max |fit - samples| exceeds this (column units, i.e. seconds).")
    ph.add_argument("--no-validate", action="store_true",
                    help="Skip the fit-quality gates (truncation convergence, --max-err).")
    ph.add_argument("--auto", action="store_true",
                    help="On gate failure, retry with higher order (up to 15) and halved/quartered granule length.")
    ph.add_argument("--out", default=None,
                    help="Output .asc path for the native product (default: CSV path with .asc "
                         "extension). The --also-wrt-tcx companion always uses the CSV stem + _tcx.asc.")
    ph.add_argument(
        "--anchor", nargs="*", type=float, metavar="JD", default=None,
        help=(
            "Zero the fitted series at an epoch instead of the integration start. "
            "Bare --anchor uses the 1977 origin (2443144 0.5003725 — all TCX read the "
            "same JD there by definition); --anchor INT FRAC uses a custom split-JD "
            "epoch (native timescale, must lie inside the CSV span). With --also-wrt-tcx "
            "the anchor also fixes the zero convention of the TCX argument labels."
        ),
    )
    ph.add_argument(
        "--also-wrt-tcx", action="store_true",
        help=(
            "Also write a second .asc re-labelled onto the local timescale TCX "
            "(exact per-event map, no interpolation). Always paired with the native "
            "product, never written alone; the two are different functions, not a "
            "constant offset apart."
        ),
    )
    ph.set_defaults(func=_cmd_chebyshev)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
