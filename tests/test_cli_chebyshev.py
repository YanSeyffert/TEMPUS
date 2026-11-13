# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_cli_chebyshev.py
# 'tempus chebyshev' subcommand: CSV -> .asc round trip, fit-quality gates, --auto.

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from tempus.chebycode import ChebyshevSeries
from tempus.cli import main
from tempus.model import jd_grid
from tempus.output import auto_out_name, read_csv, write_csv


def _make_csv(path: Path, with_rate: bool = False):
    grid = jd_grid(2451545, 0.0, 2451569, 0.0, 0.1)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    y = np.sin(0.5 * t) + 0.01 * t
    cols = {
        "JD_int": grid.jd_int,
        "JD_frac": grid.jd_frac,
        "SecondsPastJ2000": grid.et(),
        "TCB_minus_TCX": y,
    }
    if with_rate:
        dy_per_day = 0.5 * np.cos(0.5 * t) + 0.01
        cols["dTCX_dTCB"] = 1.0 - dy_per_day / 86400.0
    write_csv(path, cols, meta={"target": "TEST", "ephem_file": "fake.dat", "t_timescale": "TDB"})
    return grid, y


def test_round_trip_csv_to_asc(tmp_path):
    csv = tmp_path / "run.csv"
    grid, y = _make_csv(csv)
    assert main(["chebyshev", str(csv)]) == 0
    asc = csv.with_suffix(".asc")
    assert asc.is_file()
    cs = ChebyshevSeries.from_asc(str(asc))
    assert_allclose(cs.evaluate(grid.jd_int, grid.jd_frac), y, rtol=0.0, atol=1e-9)


def test_use_rate_with_c1_distribution(tmp_path):
    csv = tmp_path / "run.csv"
    grid, y = _make_csv(csv, with_rate=True)
    assert main(["chebyshev", str(csv), "--use-rate", "--dist", "1"]) == 0
    cs = ChebyshevSeries.from_asc(str(csv.with_suffix(".asc")))
    assert_allclose(cs.evaluate(grid.jd_int, grid.jd_frac), y, rtol=0.0, atol=1e-9)


def test_dist1_without_rate_is_an_error(tmp_path):
    csv = tmp_path / "run.csv"
    _make_csv(csv)
    assert main(["chebyshev", str(csv), "--dist", "1"]) == 2


def test_gate_failure_returns_nonzero_but_writes(tmp_path):
    csv = tmp_path / "run.csv"
    _make_csv(csv)
    assert main(["chebyshev", str(csv), "--order", "2", "--max-err", "1e-9"]) == 1
    assert csv.with_suffix(".asc").is_file()


def test_auto_escalates_until_gate_passes(tmp_path):
    csv = tmp_path / "run.csv"
    grid, y = _make_csv(csv)
    assert main(["chebyshev", str(csv), "--order", "6", "--auto", "--max-err", "1e-7"]) == 0
    cs = ChebyshevSeries.from_asc(str(csv.with_suffix(".asc")))
    assert np.max(np.abs(cs.evaluate(grid.jd_int, grid.jd_frac) - y)) <= 1e-7


def test_read_csv_round_trip(tmp_path):
    csv = tmp_path / "run.csv"
    grid, y = _make_csv(csv)
    cols, meta = read_csv(csv)
    assert meta["target"] == "TEST"
    assert cols["JD_int"].dtype == np.int64
    np.testing.assert_array_equal(cols["JD_int"], grid.jd_int)
    assert_allclose(cols["TCB_minus_TCX"], y, rtol=0.0, atol=1e-16)


def test_auto_out_name(tmp_path, monkeypatch):
    grid = jd_grid(2415020, 0.0, 2488070, 0.0, 36525.0)  # J2000-100yr .. J2000+100yr
    target = SimpleNamespace(key="EARTH_CENTER", naif=399)
    ssb = SimpleNamespace(key="SSB", naif=0)
    moon = SimpleNamespace(key="MOON", naif=301)
    earth = SimpleNamespace(key="EARTH_CENTER", naif=399)

    # Run from a scratch CWD with no sibling ../output: bare filename.
    monkeypatch.chdir(tmp_path)
    name = auto_out_name(target, ssb, grid, "/data/ephem/DE440.dat")
    assert name == "tcg_m100_p100_de440.csv"
    name = auto_out_name(moon, earth, grid, "inpop21a_TCB.dat")
    assert name == "tcg_tcl_m100_p100_inpop21a_tcb.csv"

    # INPOP distribution suffix (_m<yr>_p<yr>_<ts>) is stripped so the span and
    # timescale tag are not duplicated; the ephemeris flavour (_tcb) is kept.
    name = auto_out_name(target, ssb, grid, "inpop19a_TCB_m100_p100_tcg.dat")
    assert name == "tcg_m100_p100_inpop19a_tcb.csv"
    name = auto_out_name(target, ssb, grid, "inpop21a_TDB_m1000_p1000_tt.dat")
    assert name == "tcg_m100_p100_inpop21a_tdb.csv"

    # A span starting exactly at J2000 tags the start as "m" (before-side
    # boundary), not "p": otherwise J2000 -> J2000+100yr would read p000_p100.
    at_j2000 = jd_grid(2451545, 0.0, 2488070, 0.0, 36525.0)  # J2000 .. J2000+100yr
    name = auto_out_name(target, ssb, at_j2000, "/data/ephem/DE440.dat")
    assert name == "tcg_m000_p100_de440.csv"

    # A boundary that is only *nominally* J2000 (e.g. TT->TCB conversion of a
    # J2000 TT epoch lands a hair past JD 2451545.0 from the TCB-TT drift since
    # the 1977 origin) must still tie-break by position, not by the sign of
    # that tiny residual.
    near_j2000 = jd_grid(2451545, 1.3e-4, 2488070, -1.3e-4, 36525.0)
    name = auto_out_name(target, ssb, near_j2000, "/data/ephem/DE440.dat")
    assert name == "tcg_m000_p000_de440.csv"

    # A perturbing asteroid file adds an _ast tag before the extension.
    name = auto_out_name(target, ssb, grid, "/data/ephem/DE440.dat",
                          asteroid_file="../data/ephem/sim20201224Tyref13quar.dat")
    assert name == "tcg_m100_p100_de440_ast.csv"

    # With a sibling ../output directory, the file is routed into it.
    work = tmp_path / "work"
    (tmp_path / "output").mkdir()
    work.mkdir()
    monkeypatch.chdir(work)
    routed = auto_out_name(target, ssb, grid, "/data/ephem/DE440.dat")
    assert Path(routed) == Path("..") / "output" / "tcg_m100_p100_de440.csv"

    # Run from the project root itself, its own output/ directory is used.
    monkeypatch.chdir(tmp_path)
    routed = auto_out_name(target, ssb, grid, "/data/ephem/DE440.dat")
    assert Path(routed) == Path("output") / "tcg_m100_p100_de440.csv"
