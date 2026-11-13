# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# GM lookup from ephemeris constants (asteroid GMs: test_ephem_asteroid_gm.py).
#
# - Unit detection reads the Sun GM key of each ephemeris family.
# - GM of the Earth-Moon barycenter (NAIF 3), needed by "--ext auto" for any
#   target other than Earth or Moon. INPOP and DE files carry an EMB key; EPM
#   .tpc files only give Earth and Moon, so the GM is their sum.

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Minimal calcephpy.Constants stub, only when the real package is missing
# (same guard as test_ephem_asteroid_gm.py).
if "calcephpy" not in sys.modules:
    try:
        import calcephpy  # noqa: F401
    except ImportError:
        m = types.ModuleType("calcephpy")
        m.Constants = types.SimpleNamespace(ASTEROID=2_000_000)
        sys.modules["calcephpy"] = m

from tempus.ephem import AU3_PER_DAY2_TO_M3_PER_S2, KM3_PER_S2_TO_M3_PER_S2, CalcephEphem

NAIF_EMB = 3


class _DummyEph:
    def __init__(self, constants: dict[str, float]):
        self._items = list(constants.items())

    def getconstantcount(self):
        return len(self._items)

    def getconstantindex(self, i):
        return self._items[i - 1]


def _eph(constants: dict[str, float]) -> CalcephEphem:
    return CalcephEphem(path=Path("."), eph=_DummyEph(constants))


@pytest.mark.parametrize("sun_key, sun_value, unit", [
    ("GM_SUN",    2.959122082855911e-4, "AU3/DAY2"),  # INPOP
    ("GMS",       2.959122082855911e-4, "AU3/DAY2"),  # DE header
    ("BODY10_GM", 1.327124400419394e11, "KM3/S2"),    # SPICE .tpc (DE, EPM)
])
def test_gm_unit_read_from_sun_key(sun_key, sun_value, unit):
    # A GM-like value >= 1 that is not the Sun: the magnitude fallback
    # (largest GM-like value) would wrongly answer KM3/S2 for the AU cases.
    eph = _eph({sun_key: sun_value, "BODY399_GM": 1.0})
    assert eph.detect_gm_unit() == unit


def test_emb_gm_from_earth_plus_moon_epm_style():
    # Values from epm2021.tpc (km^3/s^2), which has no BODY3_GM.
    eph = _eph({
        "BODY10_GM": 1.327124400431652e11,
        "BODY399_GM": 3.986004362929533e05,
        "BODY301_GM": 4.902800088050693e03,
    })
    expected = (3.986004362929533e05 + 4.902800088050693e03) * KM3_PER_S2_TO_M3_PER_S2
    assert eph.gm_m3_s2(NAIF_EMB) == expected


@pytest.mark.parametrize("emb_key, emb_value, sun_key, sun_value, to_si", [
    ("GM_EMB",   8.997011390199871e-10, "GM_SUN",    2.959122082855911e-4, AU3_PER_DAY2_TO_M3_PER_S2),  # INPOP
    ("BODY3_GM", 4.035032356254802e05,  "BODY10_GM", 1.327124400419394e11, KM3_PER_S2_TO_M3_PER_S2),   # DE
])
def test_emb_key_takes_priority_over_earth_plus_moon(emb_key, emb_value, sun_key, sun_value, to_si):
    # Earth and Moon set to values that do not sum to the EMB key, so the
    # assertion fails if the Earth+Moon fallback were used instead.
    eph = _eph({
        sun_key: sun_value,
        emb_key: emb_value,
        "BODY399_GM": emb_value / 4,
        "BODY301_GM": emb_value / 4,
    })
    assert eph.gm_m3_s2(NAIF_EMB) == emb_value * to_si


def test_emb_gm_missing_everywhere_still_raises():
    eph = _eph({"BODY10_GM": 1.327124400431652e11, "BODY399_GM": 3.986004362929533e05})
    with pytest.raises(KeyError, match="NAIF 3"):
        eph.gm_m3_s2(NAIF_EMB)
