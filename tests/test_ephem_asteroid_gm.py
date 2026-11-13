# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Provide a minimal calcephpy.Constants stub for environments without calcephpy.
# Only when the real package is genuinely missing: registering it unconditionally
# would shadow a real install for every test importing calcephpy afterwards.
if "calcephpy" not in sys.modules:
    try:
        import calcephpy  # noqa: F401
    except ImportError:
        m = types.ModuleType("calcephpy")
        m.Constants = types.SimpleNamespace(ASTEROID=2_000_000)
        sys.modules["calcephpy"] = m

from tempus.ephem import AU3_PER_DAY2_TO_M3_PER_S2, CalcephEphem


class _DummyEph:
    def __init__(self, constants: dict[str, float]):
        self._items = list(constants.items())

    def getconstantcount(self):
        return len(self._items)

    def getconstantindex(self, i):
        return self._items[i - 1]


class _DummyAstEph:
    def __init__(self, records):
        self._records = list(records)

    def getpositionrecordcount(self):
        return len(self._records)

    def getpositionrecordindex(self, i):
        return self._records[i - 1]


def test_asteroid_index_reverse_mapping_base():
    assert CalcephEphem._asteroid_index_from_object_id(2_000_001) == 1
    assert CalcephEphem._asteroid_index_from_object_id(12345) is None


def test_asteroid_object_id_uses_base():
    assert CalcephEphem.asteroid_object_id(1) == 2_000_001
    assert CalcephEphem.asteroid_object_id(324) == 2_000_324


def test_asteroid_index_from_any_accepts_both_conventions():
    # What --asteroid-ids relies on (Orcus: index 90482, NAIF SPK ID 2090482).
    assert CalcephEphem.asteroid_index_from_any(2_090_482) == 90482
    assert CalcephEphem.asteroid_index_from_any(90482) == 90482


def test_asteroid_indices_from_object_ids_filters_non_asteroids():
    out = CalcephEphem.asteroid_indices_from_object_ids([2_000_001, 2_000_324, 10, 17])
    assert out == [1, 324]


def test_asteroid_object_ids_in_file_discovers_and_deduplicates():
    ast = _DummyAstEph(
        [
            (2_000_001, 0, 0.0, 1.0, 1),
            (2_000_324, 0, 0.0, 1.0, 1),
            (2_000_001, 0, 1.0, 2.0, 1),
            (10, 0, 0.0, 1.0, 1),
            (17, 0, 0.0, 1.0, 1),
        ]
    )
    eph = CalcephEphem(path=Path("."), eph=_DummyEph({"GM_SUN": 2.959122128735587e-4}), asteroid_eph=ast)
    assert eph.asteroid_object_ids_in_file() == [2_000_001, 2_000_324]


def test_gm_m3_s2_reads_inpop_ma_constants_for_asteroid():
    eph = CalcephEphem(
        path=Path("."),
        eph=_DummyEph(
            {
                # Needed for detect_gm_unit() in the shared conversion path.
                "GM_SUN": 2.959122128735587e-4,
                "MA0001": 1.3964353459423487e-13,
            }
        ),
    )
    gm = eph.gm_m3_s2(2_000_001)
    assert gm == 1.3964353459423487e-13 * AU3_PER_DAY2_TO_M3_PER_S2


def test_asteroid_indices_with_gm_in_main_file_reads_ma_keys():
    eph = CalcephEphem(
        path=Path("."),
        eph=_DummyEph(
            {
                "GM_SUN": 2.959122128735587e-4,
                "MA0001": 1.0e-13,
                "MA0324": 2.0e-13,
                "MA12345": 3.0e-13,
                "GM_MAR": 9.0e-11,
            }
        ),
    )
    assert eph.asteroid_indices_with_gm_in_main_file() == [1, 324, 12345]
