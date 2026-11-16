# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_arg_timescale.py
# Argument-timescale re-parameterisation (Klioner 2026) via the exact per-sample
# event map, exposed as `tempus chebyshev --also-wrt-tcx`.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tempus.chebycode import ChebyshevSeries
from tempus.cli import main
from tempus.model import (
    _T0_FRAC,
    _T0_INT,
    DAY_S,
    L_B,
    TDB0_S,
    TimeGrid,
    jd_grid,
    relabel_to_tcx,
)
from tempus.output import write_csv


def _write_csv(path, grid, y):
    write_csv(path, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                     "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": y}, meta={})


def test_exact_map_linear_closed_form(tmp_path):
    # Constant rate k: y(t) = k*(t-t0)*86400  =>  y_wrt_T(T) = 86400*k*(T-t0)/(1-k).
    k = 2.0e-8
    grid = jd_grid(2451545, 0.0, 2451645, 0.0, 1.0)
    t = (grid.jd_int - 2451545).astype(np.float64) + grid.jd_frac
    csv = tmp_path / "run.csv"
    _write_csv(csv, grid, k * t * float(DAY_S))

    assert main(["chebyshev", str(csv), "--also-wrt-tcx",
                 "--order", "5", "--granule-len", "8", "--max-err", "1e-9"]) == 0
    cs = ChebyshevSeries.from_asc(str(tmp_path / "run_tcx.asc"))

    Tq = jd_grid(2451550, 0.25, 2451640, 0.25, 5.0)
    d_t0 = (Tq.jd_int - 2451545).astype(np.float64) + Tq.jd_frac
    assert_allclose(cs.evaluate(Tq.jd_int, Tq.jd_frac),
                    float(DAY_S) * k * d_t0 / (1.0 - k), rtol=0.0, atol=1e-9)


def test_exact_map_identity_curved_offset(tmp_path):
    # Every sample is an event carrying both labels: the TCX series evaluated at
    # T_k = t_k - y_k/86400 must return y_k (Klioner's consistency identity).
    grid = jd_grid(2451545, 0.0, 2451745, 0.0, 0.5)
    t = (grid.jd_int - 2451545).astype(np.float64) + grid.jd_frac
    y = 1.4e-8 * t * float(DAY_S) + 0.5 * np.sin(2.0 * np.pi * t / 27.3)
    csv = tmp_path / "run.csv"
    _write_csv(csv, grid, y)

    assert main(["chebyshev", str(csv), "--also-wrt-tcx",
                 "--order", "11", "--granule-len", "4", "--max-err", "1e-6"]) == 0
    cs = ChebyshevSeries.from_asc(str(tmp_path / "run_tcx.asc"))

    T = relabel_to_tcx(TimeGrid(grid.jd_int, grid.jd_frac), y)
    assert np.max(np.abs(cs.evaluate(T.jd_int, T.jd_frac) - y)) < 1e-9


def _days_since_t0(g: TimeGrid) -> np.ndarray:
    # Cancellation-free: the integer part is exact, and both fracs are O(1).
    return (g.jd_int - _T0_INT).astype(np.float64) + (g.jd_frac - _T0_FRAC)


def _y_wrt_tcb(g: TimeGrid) -> np.ndarray:
    # y = TCB - TDB [s] as a function of the TCB label, straight from the
    # IAU 2006 B3 defining relation.  Never differences two nearby epochs.
    return L_B * _days_since_t0(g) * float(DAY_S) - TDB0_S


def _y_wrt_tdb(g: TimeGrid) -> np.ndarray:
    # Same offset as a function of the TDB label.  Substituting
    # d_t0(TCB) = d_t0(TDB) + y/86400 into the relation and solving for y
    # gives the exact inverse; it is still linear in the epoch.
    return (L_B * _days_since_t0(g) * float(DAY_S) - TDB0_S) / (1.0 - L_B)


def test_cli_tdb_closed_form(tmp_path):
    # Flagship: y = TCB - TDB on a TCB grid (exact IAU 2006 B3 relation).
    #
    # Both reference values come from the closed form above rather than from
    # differencing split JDs: a difference of jd_frac scaled by 86400 carries
    # 86400*2**-53 ~ 1e-11 s of quantisation, which would swamp everything
    # measured here.  The closed form is good to ~2 ulp of y (~4e-15 s).
    grid = jd_grid(2451545, 0.0, 2451585, 0.0, 0.5)
    y = _y_wrt_tcb(grid)

    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": y},
              meta={"target": "TDB-TEST", "in_timescale": "TCB"})

    rc = main(["chebyshev", str(csv), "--also-wrt-tcx",
               "--order", "5", "--granule-len", "8", "--max-err", "1e-9"])
    assert rc == 0
    asc = tmp_path / "run_tcx.asc"
    assert asc.is_file()

    # The native series fits an exactly-linear function on granule edges that
    # are whole days, so the JD->tau reduction is cancellation-free and this
    # holds at float precision: ~4 ulp of y ~ 11.3 s.
    native = ChebyshevSeries.from_asc(str(tmp_path / "run.asc"))
    assert_allclose(native.evaluate(grid.jd_int, grid.jd_frac), y,
                    rtol=0.0, atol=2e-14)

    # The TCX series cannot reach float precision, and the limit is the .asc
    # format, not the re-parameterisation: granule edges are stored as a single
    # float64 JD (~5e-10 day of resolution near J2000) while sample abscissae
    # are split int/frac.  Relabelling shifts every edge by y/86400 ~ 1.3e-4 d
    # off a whole day, so that resolution now bites; times the linear
    # coefficient (~5.4e-3 s) it costs ~3e-13 s.  Measured: 2.5e-13 s.
    cs = ChebyshevSeries.from_asc(str(asc))
    T_test = jd_grid(2451546, 0.25, 2451584, 0.25, 1.0)  # TDB epochs inside the span
    assert_allclose(cs.evaluate(T_test.jd_int, T_test.jd_frac),
                    _y_wrt_tdb(T_test), rtol=0.0, atol=1e-12)


def test_anchor_zeroes_series_at_epoch(tmp_path):
    grid = jd_grid(2451545, 0.0, 2451569, 0.0, 0.1)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    y = np.sin(0.5 * t) + 3.0
    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": y}, meta={})
    assert main(["chebyshev", str(csv), "--anchor", "2451557", "0.0", "--max-err", "1e-8"]) == 0
    cs = ChebyshevSeries.from_asc(str(csv.with_suffix(".asc")))
    val = cs.evaluate(np.array([2451557], dtype=np.int64), np.array([0.0]))[0]
    assert abs(val) < 1e-9


def test_anchor_1977_outside_span_errors(tmp_path):
    grid = jd_grid(2451545, 0.0, 2451569, 0.0, 0.1)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": np.sin(0.5 * t)},
              meta={"in_timescale": "TCB"})
    assert main(["chebyshev", str(csv), "--anchor", "--max-err", "1e-8"]) == 2


@pytest.mark.parametrize("timescale", ["TCB", "TDB"])
def test_bare_anchor_places_1977_origin_on_the_csv_axis(tmp_path, timescale):
    # y = 1 s/day * (t - grid start): the anchored fit is exactly linear, so its
    # value at an epoch measures that epoch's distance from the zero in days.
    grid = jd_grid(2443140, 0.0, 2443150, 0.0, 0.25)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": t},
              meta={"in_timescale": timescale})
    assert main(["chebyshev", str(csv), "--anchor", "--max-err", "1e-10"]) == 0
    asc = csv.with_suffix(".asc")
    cs = ChebyshevSeries.from_asc(str(asc))

    tdb0_days = TDB0_S / DAY_S                      # the origin's TDB date is T0 + TDB0/86400
    zero_frac = _T0_FRAC + (tdb0_days if timescale == "TDB" else 0.0)
    at = lambda frac: cs.evaluate(np.array([_T0_INT], dtype=np.int64), np.array([frac]))[0]
    assert abs(at(zero_frac)) < 1e-12
    if timescale == "TDB":
        assert_allclose(at(_T0_FRAC), -tdb0_days, rtol=0.0, atol=1e-14)   # T0 itself is 65.5 us later
    title = asc.read_text().splitlines()[0]
    assert f"argument_timescale={timescale}" in title
    assert f"zero={timescale}:JD{_T0_INT}+{float(zero_frac)!r}(IAU1977)" in title


def test_bare_anchor_without_timescale_errors(tmp_path):
    grid = jd_grid(2443140, 0.0, 2443150, 0.0, 0.25)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": t}, meta={})
    assert main(["chebyshev", str(csv), "--anchor", "--max-err", "1e-10"]) == 2


def test_anchor_refuses_non_integration_column(tmp_path):
    # A column with a definitional zero (e.g. TT_minus_TDB) must never be
    # anchored, even if it's the only data column present. anchorable_column
    # in the metadata records what integrate actually left as a free
    # constant; --anchor must refuse anything else.
    grid = jd_grid(2451545, 0.0, 2451569, 0.0, 0.1)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TT_minus_TDB": np.sin(0.5 * t)},
              meta={"anchorable_column": "TCB_minus_TCX"})
    rc = main(["chebyshev", str(csv), "--column", "TT_minus_TDB",
               "--anchor", "2451557", "0.0", "--max-err", "1e-8"])
    assert rc == 2


def test_cli_native_default_unchanged(tmp_path):
    # Without the flag the output name and argument stay native.
    grid = jd_grid(2451545, 0.0, 2451569, 0.0, 0.1)
    t = (grid.jd_int - grid.jd_int[0]).astype(np.float64) + grid.jd_frac
    y = np.sin(0.5 * t)
    csv = tmp_path / "run.csv"
    write_csv(csv, {"JD_int": grid.jd_int, "JD_frac": grid.jd_frac,
                    "SecondsPastJ2000": grid.et(), "TCB_minus_TCX": y}, meta={})
    assert main(["chebyshev", str(csv), "--max-err", "1e-8"]) == 0
    assert (tmp_path / "run.asc").is_file()
    assert not (tmp_path / "run_tcx.asc").is_file()
