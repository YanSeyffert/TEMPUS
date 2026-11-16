# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_zero.py
# integrate --zero: the 1977 origin on each axis, the local-polynomial value at
# an epoch, and end-to-end runs (skipped where the ephemeris files are not
# available).

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tempus.cli import main
from tempus.model import DAY_S, TDB0_S, TimeScalar, jd_grid, jd_origin_1977, value_at_epoch

ROOT = Path(__file__).resolve().parents[1]
EPH = {"TDB": ROOT / "data/ephem/inpop21a_TDB_m100_p100_tt.dat",
       "TCB": ROOT / "data/ephem/inpop21a_TCB_m100_p100_tcg.dat"}
needs_tdb = pytest.mark.skipif(not EPH["TDB"].is_file(), reason="ephemeris data file not available")


def _integrate(tmp_path, name, eph, *extra, step="0.75", center="SSB"):
    """Moon (or a difference with --center), 50 days around the 1977 origin."""
    csv = tmp_path / f"{name}.csv"
    assert main(["integrate", "--eph-file", str(eph), "--jd-start", "2443120", "0.5",
                 "--jd-end", "2443170", "0.5", "--jd-step", step, "--target", "MOON",
                 "--center", center, "--out", str(csv), *extra]) == 0
    return np.loadtxt(csv, delimiter=",", skiprows=1), json.loads(csv.with_suffix(".run.json").read_text())


def test_origin_1977_on_each_axis():
    assert tuple(jd_origin_1977("TCB")) == (2443144, 0.5003725)
    tdb = jd_origin_1977("TDB")
    assert tdb.int_part == 2443144
    assert_allclose((tdb.frac_part - 0.5003725) * DAY_S, TDB0_S, rtol=0.0, atol=1e-11)   # float64 step of the fraction: 1e-11 s
    with pytest.raises(ValueError):
        jd_origin_1977("TT")


def test_value_at_epoch_reproduces_a_polynomial_and_stays_inside():
    grid = jd_grid(2443100, 0.5, 2443190, 0.5, 0.75)
    t = (grid.jd_int - 2443144).astype(np.float64) + (grid.jd_frac - 0.5)
    y = 3.0 + 1e-3 * t + 2e-9 * t**3                     # degree 3: reproduced exactly by 12 points
    at = TimeScalar(2443144, np.float64(0.5003725))
    x = 0.0003725
    assert_allclose(value_at_epoch(grid.jd_int, grid.jd_frac, y, at), 3.0 + 1e-3 * x + 2e-9 * x**3,
                    rtol=0.0, atol=4e-15)
    with pytest.raises(ValueError):                       # outside the sampled span
        value_at_epoch(grid.jd_int, grid.jd_frac, y, TimeScalar(2443300, np.float64(0.5)))
    with pytest.raises(ValueError):                       # fewer samples than points
        value_at_epoch(grid.jd_int[:5], grid.jd_frac[:5], y[:5], at)


@pytest.mark.parametrize("ts", ["TDB", "TCB"])
def test_zero_at_the_1977_origin(tmp_path, ts):
    if not EPH[ts].is_file():
        pytest.skip("ephemeris data file not available")
    raw, _ = _integrate(tmp_path, "raw", EPH[ts])
    zeroed, meta = _integrate(tmp_path, "zeroed", EPH[ts], "--zero")
    origin = jd_origin_1977(ts)
    assert meta["zero"]["epoch"] == [origin.int_part, float(origin.frac_part)]
    assert meta["zero"]["timescale"] == ts and meta["zero"]["origin"] == "IAU 1977"
    # the same run, moved by one constant, and 0 at the origin
    assert_allclose(zeroed[:, 3] - raw[:, 3], meta["zero"]["shift_s"], rtol=0.0, atol=1e-17)
    assert abs(value_at_epoch(zeroed[:, 0].astype(np.int64), zeroed[:, 1], zeroed[:, 3], origin)) < 2e-16


@needs_tdb
def test_run_without_zero_records_the_integration_start(tmp_path):
    raw, meta = _integrate(tmp_path, "raw", EPH["TDB"])
    assert raw[0, 3] == 0.0
    assert meta["zero"] == {"epoch": [2443120, 0.5], "timescale": "TDB", "origin": "integration start",
                            "method": "first sample", "shift_s": 0.0}
    assert main(["chebyshev", str(tmp_path / "raw.csv"), "--max-err", "1e-10"]) == 0
    assert "zero=TDB:JD2443120+0.5(custom)" in (tmp_path / "raw.asc").read_text().splitlines()[0]


@needs_tdb
def test_chebyshev_records_the_zero_of_a_zeroed_run(tmp_path):
    _integrate(tmp_path, "zeroed", EPH["TDB"], "--zero")
    assert main(["chebyshev", str(tmp_path / "zeroed.csv"), "--max-err", "1e-10"]) == 0
    origin = jd_origin_1977("TDB")
    title = (tmp_path / "zeroed.asc").read_text().splitlines()[0]
    assert f"zero=TDB:JD{origin.int_part}+{float(origin.frac_part)!r}(IAU1977)" in title


@needs_tdb
def test_zero_agrees_with_a_fine_step_run(tmp_path):
    # Zeroed 18-h and 1.125-h runs agree to the 18-h run's own integration
    # error: 3e-12 s for the Moon in this window.
    coarse, _ = _integrate(tmp_path, "coarse", EPH["TDB"], "--zero")
    fine, _ = _integrate(tmp_path, "fine", EPH["TDB"], "--zero", step=repr(0.75 / 16))
    assert_allclose(coarse[:, 3], fine[::16, 3], rtol=0.0, atol=5e-12)


@needs_tdb
def test_zero_at_a_custom_epoch_and_outside_the_grid(tmp_path):
    zeroed, meta = _integrate(tmp_path, "custom", EPH["TDB"], "--zero", "2443150", "0.5")
    assert meta["zero"]["origin"] == "custom" and meta["zero"]["epoch"] == [2443150, 0.5]
    node = np.nonzero((zeroed[:, 0] == 2443150) & (zeroed[:, 1] == 0.5))[0][0]
    assert abs(zeroed[node, 3]) < 2e-16                   # on a grid node: that sample is 0

    # an epoch typed in that is the 1977 origin is recognised as such
    _, meta = _integrate(tmp_path, "typed", EPH["TDB"], "--zero", "2443144", "0.5003724992418982")
    assert meta["zero"]["origin"] == "IAU 1977"


@needs_tdb
def test_zero_outside_the_grid_stops_before_integrating(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tempus.cli.Delta_TCX", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("integrated although the --zero epoch lies outside the grid")))
    csv = tmp_path / "outside.csv"                        # a 2000 run does not contain the 1977 origin
    assert main(["integrate", "--eph-file", str(EPH["TDB"]), "--jd-start", "2451545", "0.0",
                 "--jd-end", "2451560", "0.0", "--jd-step", "0.75", "--target", "MOON",
                 "--center", "SSB", "--zero", "--out", str(csv)]) == 2
    err = capsys.readouterr().err
    assert "the 1977 origin" in err and "omit --zero" in err and "integrated although" not in err
    assert not csv.exists()


@needs_tdb
def test_zero_in_difference_mode(tmp_path):
    zeroed, meta = _integrate(tmp_path, "diff", EPH["TDB"], "--zero", center="EARTH_CENTER")
    assert meta["tempus_integrated_column"] == "TCG_minus_TCL"
    origin = jd_origin_1977("TDB")
    assert abs(value_at_epoch(zeroed[:, 0].astype(np.int64), zeroed[:, 1], zeroed[:, 3], origin)) < 2e-16
