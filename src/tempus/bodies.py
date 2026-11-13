# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/bodies.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Canonical major-body NAIF IDs (barycenters + Sun + SSB)
NAIF: Dict[str, int] = {
    "SSB": 0,
    "SUN": 10,
    "MERCURY_BARYCENTER": 1,
    "VENUS_BARYCENTER": 2,
    "EARTH_MOON_BARYCENTER": 3,
    "EARTH_CENTER": 399,
    "MOON": 301,
    "MARS_BARYCENTER": 4,
    "JUPITER_BARYCENTER": 5,
    "SATURN_BARYCENTER": 6,
    "URANUS_BARYCENTER": 7,
    "NEPTUNE_BARYCENTER": 8,
    "PLUTO_BARYCENTER": 9,
    "Ceres": 2000001,
    "Pallas": 2000002,
    "Vesta": 2000004,
    "Iris": 2000007,
    "Bamberga": 2000324,
    "Hygiea": 2000010,
}

DEFAULT_EXT_KEYS: List[str] = [
    "SUN",
    "MERCURY_BARYCENTER",
    "VENUS_BARYCENTER",
    "EARTH_MOON_BARYCENTER",
    "MARS_BARYCENTER",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
    "PLUTO_BARYCENTER",
]

_INV_NAIF: Dict[int, str] = {v: k for k, v in NAIF.items()}


def _norm_key(s: str) -> str:
    return s.strip().upper().replace(" ", "_")


def _to_spice_name(key: str) -> str:
    # SPICE: "MARS BARYCENTER", "EARTH BARYCENTER", "SUN", "SSB"
    if key == "SSB":
        return "SSB"
    if key.endswith("_BARYCENTER"):
        return key.replace("_", " ")
    return key.replace("_", " ")


@dataclass(frozen=True)
class BodyRef:
    """
    Backend-agnostic body reference.

    - naif: integer ID (preferred internal canonical)
    - key: canonical key (e.g., "MARS_BARYCENTER")
    - spice: SPICE name (e.g., "MARS BARYCENTER")
    """
    naif: int
    key: str
    spice: str

    @classmethod
    def from_naif(cls, naif_id: int) -> "BodyRef":
        key = _INV_NAIF.get(int(naif_id), None)
        if key is None:
            # Allow arbitrary numeric IDs (e.g. asteroid/object IDs) without
            # forcing registry updates in NAIF map.
            i = int(naif_id)
            return cls(naif=i, key=f"NAIF_{i}", spice=str(i))
        return cls(naif=int(naif_id), key=key, spice=_to_spice_name(key))

    @classmethod
    def from_key(cls, key: str) -> "BodyRef":
        k = _norm_key(key)
        if k not in NAIF:
            raise ValueError(f"Unknown body '{key}'. Add it to bodies.NAIF.")
        return cls(naif=NAIF[k], key=k, spice=_to_spice_name(k))


def parse_body(x: Union[int, str, BodyRef]) -> BodyRef:
    """
    Accepts:
      - int: 4
      - str: "4", "MARS_BARYCENTER", "Mars barycenter", "MARS BARYCENTER"
      - BodyRef: returns as-is
    """
    if isinstance(x, BodyRef):
        return x
    if isinstance(x, int):
        return BodyRef.from_naif(x)
    s = str(x).strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return BodyRef.from_naif(int(s))
    return BodyRef.from_key(s)


def ext_list(target: Union[int, str, BodyRef], base_keys: Optional[List[str]] = None) -> List[BodyRef]:
    """
    Returns auto external bodies for U_ext.

    - Uses DEFAULT_EXT_KEYS unless base_keys is provided.
    - Excludes target from the final list.
    - If target is Earth or Moon, swaps EMB with Moon or Earth respectively.
    """
    t = parse_body(target)
    replacement = None
    if t.naif == NAIF["EARTH_CENTER"]:
        replacement = "MOON"
    elif t.naif == NAIF["MOON"]:
        replacement = "EARTH_CENTER"

    keys: List[str] = []
    for raw in (base_keys or DEFAULT_EXT_KEYS):
        k = _norm_key(raw)
        if replacement is not None and k == "EARTH_MOON_BARYCENTER":
            keys.append(replacement)
        else:
            keys.append(k)

    refs = [BodyRef.from_key(k) for k in keys]
    return [r for r in refs if r.naif != t.naif]
