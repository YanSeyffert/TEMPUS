# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_cli_smoke.py
# End-to-end CLI: a tiny real integrate run -> chebyshev, exercising the parts
# the library-level tests never touch (arg parsing, ephemeris open, file names,
# run metadata). Skipped where the ephemeris data files are not available.

import json
from pathlib import Path

import pytest

from tempus.cli import main

ROOT = Path(__file__).resolve().parents[1]
EPH = ROOT / "data/ephem/inpop21a_TDB_m100_p100_tt.dat"

pytestmark = pytest.mark.skipif(not EPH.is_file(), reason="ephemeris data file not available")


def test_integrate_then_chebyshev(tmp_path):
    csv = tmp_path / "smoke.csv"
    assert main(["integrate", "--eph-file", str(EPH),
                 "--jd-start", "2451545", "0.0", "--jd-end", "2451560", "0.0",
                 "--jd-step", "0.75", "--target", "MARS_BARYCENTER", "--center", "SSB",
                 "--out", str(csv)]) == 0

    meta = json.loads(csv.with_suffix(".run.json").read_text())
    assert meta["tempus_integrated_column"] == "TCB_minus_TCX"
    assert meta["tempus_version"] and meta["in_timescale"] == "TDB"

    assert main(["chebyshev", str(csv), "--max-err", "1e-9"]) == 0
    assert "argument_timescale=" in (tmp_path / "smoke.asc").read_text().splitlines()[0]


def test_non_ssb_center_records_both_legs(tmp_path):
    csv = tmp_path / "diff.csv"
    assert main(["integrate", "--eph-file", str(EPH),
                 "--jd-start", "2451545", "0.0", "--jd-end", "2451550", "0.0",
                 "--jd-step", "0.75", "--target", "MOON", "--center", "EARTH_CENTER",
                 "--out", str(csv)]) == 0

    meta = json.loads(csv.with_suffix(".run.json").read_text())
    assert meta["tempus_integrated_column"] == "TCG_minus_TCL"
    assert meta["simultaneity"] == "TCB"
    # Each leg excludes its own body and gains the other as a perturber.
    assert "EARTH_CENTER" in meta["ext_resolved_target"]
    assert "MOON" in meta["ext_resolved_center"]
