# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_cli_missing_files.py
# 'tempus integrate' names the missing input file before opening any ephemeris.

from tempus.cli import main

GRID = ["--jd-start", "2451545", "0.0", "--jd-end", "2451546", "0.0", "--jd-step", "0.75"]


def test_missing_eph_file_is_named(tmp_path, capsys):
    missing = tmp_path / "not_downloaded.dat"
    rc = main(["integrate", "--eph-file", str(missing), *GRID, "--out", str(tmp_path / "x.csv")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--eph-file: file not found" in err and "not_downloaded.dat" in err
    assert "download_data.sh" in err


def test_missing_asteroid_file_is_named(tmp_path, capsys):
    eph = tmp_path / "present.dat"
    eph.write_text("")    # exists; the check must fail on the asteroid file before opening it
    rc = main(["integrate", "--eph-file", str(eph), *GRID, "--out", str(tmp_path / "x.csv"),
               "--asteroid-file", str(tmp_path / "sb441-n373s.bsp")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--asteroid-file: file not found" in err and "sb441-n373s.bsp" in err
