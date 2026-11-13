# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempus.bodies import ext_list, parse_body


def _keys(target):
    return [b.key for b in ext_list(target)]


def test_auto_ext_excludes_non_earth_moon_target():
    keys = _keys("MARS_BARYCENTER")
    assert "MARS_BARYCENTER" not in keys
    assert "EARTH_MOON_BARYCENTER" in keys


def test_auto_ext_replaces_emb_with_moon_for_earth_target():
    keys = _keys("EARTH_CENTER")
    assert "EARTH_MOON_BARYCENTER" not in keys
    assert "MOON" in keys
    assert "EARTH_CENTER" not in keys


def test_auto_ext_replaces_emb_with_earth_for_moon_target():
    keys = _keys("MOON")
    assert "EARTH_MOON_BARYCENTER" not in keys
    assert "EARTH_CENTER" in keys
    assert "MOON" not in keys


def test_parse_unknown_numeric_body_id_is_allowed():
    b = parse_body("2000345")
    assert b.naif == 2000345
    assert b.key == "NAIF_2000345"
