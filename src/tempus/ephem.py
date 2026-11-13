# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/ephem.py
# Ephemeris backend: CALCEPH (reads INPOP/DE/EPM binaries and SPICE SPK/PCK).
# A SpiceyPy backend existed early on and was removed once CALCEPH covered the
# same files; see docs/DEVELOPER_NOTES.md for what a backend must provide.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np

# J2000 epoch as Julian Date
JD_J2000 = 2451545.0

# Constants for unit conversions
AU_M = 149_597_870_700.0
DAY_S = 86_400.0
AU3_PER_DAY2_TO_M3_PER_S2 = (AU_M**3) / (DAY_S**2)
KM3_PER_S2_TO_M3_PER_S2 = 1e9
# AU³/day² → km³/s²  (used when supplementing header constants into a km³/s² cache)
AU3_PER_DAY2_TO_KM3_PER_S2 = AU3_PER_DAY2_TO_M3_PER_S2 / KM3_PER_S2_TO_M3_PER_S2


# -------------------------
# CALCEPH backend 
# -------------------------
@dataclass
class CalcephEphem:
    """
    CALCEPH ephemeris reader for JPL/INPOP/EPM binary ephemerides (e.g., .jpl, .tcheb).
    Uses NAIF integer IDs (e.g., SSB=0, Sun=10, Mars barycenter=4).

    Conventions:
    - Input time: seconds past J2000 in the ephemeris file's native timescale (often TDB-like).
    - Output: SI units (meters, meters/second).
    """
    path: Path
    eph: object  # calcephpy.CalcephBin
    asteroid_path: Path | None = None
    asteroid_eph: object | None = None  # optional secondary calcephpy.CalcephBin

    _const_cache: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _gm_unit: str | None = field(default=None, init=False, repr=False)  # "AU3/DAY2" or "KM3/S2"

    @classmethod
    def open(cls, path: str | Path | list[str | Path], asteroid_file: str | Path | None = None, ) -> "CalcephEphem":
        from calcephpy import CalcephBin
        paths = [path] if isinstance(path, (str, Path)) else path
        resolved = [str(Path(p).expanduser().resolve()) for p in paths]
        primary_path = Path(resolved[0])
        asteroid_path = None
        asteroid_eph = None
        if asteroid_file:
            asteroid_path = Path(asteroid_file).expanduser().resolve()
            # Open asteroid file in a separate handle for body discovery and position queries.
            # CALCEPH does not allow combining two INPOP/original DE binary files in one handle,
            # so the asteroid .dat file must NOT be added to resolved alongside the main .dat file.
            # Asteroid positions are chained manually in state_jd (asteroid->Sun + Sun->center).
            asteroid_eph = CalcephBin.open(str(asteroid_path))
        return cls(
            path=primary_path,
            eph=CalcephBin.open(resolved),
            asteroid_path=asteroid_path,
            asteroid_eph=asteroid_eph,
        )

    def close(self) -> None:
        self.eph.close()
        if self.asteroid_eph is not None:
            self.asteroid_eph.close()

    def timescale(self) -> str | None:
        """
        Returns CALCEPH timescale if available (useful for logging and eph scaling).
        """
        try:
            ts_enum = self.eph.gettimescale()
            ts_label = {1: "TDB", 2: "TCB"}.get(ts_enum, "unknown")
            return ts_label
        except Exception:
            return None
        
    def timescale_enum(self) -> int | None:
        """
        Returns CALCEPH timescale enum if available:
        - 1: TDB-like
        - 2: TCB-like
        - None: unknown / error
        """
        try:
            ts_enum = self.eph.gettimescale()
            return int(ts_enum)
        except Exception:
            return None
        
    @staticmethod
    def _as_jd_parts(jd_int: float | int | np.ndarray, jd_frac: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Normalize split JD input to aligned 1D arrays.
        Returns (jd_int, jd_frac, was_scalar).
        """
        ji = np.asarray(jd_int, dtype=np.int64)
        jf = np.asarray(jd_frac, dtype=np.float64)
        was_scalar = ji.ndim == 0 and jf.ndim == 0
        ji = ji.reshape(-1)
        jf = jf.reshape(-1)
        if ji.size != jf.size:
            if ji.size == 1:
                ji = np.full(jf.shape, ji[0], dtype=np.int64)
            elif jf.size == 1:
                jf = np.full(ji.shape, jf[0], dtype=np.float64)
            else:
                raise ValueError("jd_int and jd_frac must have the same length or be broadcastable.")
        return ji, jf, was_scalar

    def state_jd(self, jd_int: float | int | np.ndarray, jd_frac: float | np.ndarray, target: int, center: int = 0) -> np.ndarray:
        """
        Return state vectors for split Julian-date input.
        """
        from calcephpy import Constants

        jd0, dt, was_scalar = self._as_jd_parts(jd_int, jd_frac)

        unit = Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID

        NAIF_SUN = 10
        asteroid_base = int(Constants.ASTEROID)
        if target >= asteroid_base and self.asteroid_eph is not None:
            # INPOP/DE asteroid files store positions relative to the Sun, and CALCEPH
            # cannot combine two INPOP binary files in one handle.  Chain manually:
            #   pos(asteroid, center) = pos(asteroid, Sun) + pos(Sun, center)
            pv_ast = self.asteroid_eph.compute_unit(jd0, dt, target, NAIF_SUN, unit)
            pv_sun = self.eph.compute_unit(jd0, dt, NAIF_SUN, center, unit)
            pv = [pv_ast[i] + pv_sun[i] for i in range(6)]
        else:
            pv = self.eph.compute_unit(jd0, dt, target, center, unit)

        # pv is a list/tuple of arrays [x,y,z,vx,vy,vz] in km and km/s
        x_km = np.vstack([pv[0], pv[1], pv[2]]).T
        v_km_s = np.vstack([pv[3], pv[4], pv[5]]).T

        out = np.hstack([x_km * 1e3, v_km_s * 1e3])  # meters, m/s
        return out[0] if was_scalar else out

    
    # ----- Functions to get GM of a body in SI units -----
    def _constants_from_handle(self, handle: object) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            n = handle.getconstantcount()
        except Exception:
            return out
        for i in range(1, n + 1):
            try:
                name, val = handle.getconstantindex(i)
            except Exception:
                # EPM ephemerides report a constant count but do not support
                # getconstantindex; bail out with whatever we have so far.
                break
            if isinstance(name, str) and isinstance(val, (int, float)):
                out[name.strip().upper()] = float(val)
        return out

    def constants(self) -> dict[str, float]:
        """
        Return a dict of ephemeris header constants (uppercased keys), cached.
        Falls back to a sibling .tpc file (same stem as self.path) when
        getconstantindex is not supported by the ephemeris format (e.g. EPM).
        """
        if self._const_cache:
            return self._const_cache

        out = self._constants_from_handle(self.eph)

        if not out:
            tpc = self.path.with_suffix('.tpc')
            if tpc.exists():
                out = parse_spice_tpc(tpc)

        if self.asteroid_eph is not None:
            # Keep main-file constants authoritative on collisions.
            aux = self._constants_from_handle(self.asteroid_eph)
            aux.update(out)
            out = aux

        self._const_cache = out
        return out

    def detect_gm_unit(self) -> str:
        """
        Detect whether GM constants are stored as AU^3/day^2 or km^3/s^2.
        Uses the Sun GM constant if present (GM_SUN, GMS or BODY10_GM).
        """
        if self._gm_unit is not None:
            return self._gm_unit

        c = self.constants()

        # Sun GM key: INPOP, then DE header, then SPICE .tpc style
        sun_candidates = ["GM_SUN", "GMS", "BODY10_GM", "GM10"]
        sun_val = None
        for k in sun_candidates:
            if k in c:
                sun_val = c[k]
                break

        if sun_val is None:
            # Fallback heuristic: look for any GM-like values and decide by magnitude.
            # AU^3/day^2 typical scale: <=1e-3 for Sun, planets much smaller.
            # km^3/s^2 typical scale: 1e11 for Sun.
            gm_vals = [v for k, v in c.items() if "GM" in k]
            if not gm_vals:
                raise KeyError("No GM-like constants found in ephemeris header.")
            sun_like = max(gm_vals)
            sun_val = sun_like

        # Magnitude-based discrimination
        if sun_val < 1.0:
            self._gm_unit = "AU3/DAY2"
        else:
            self._gm_unit = "KM3/S2"
        return self._gm_unit

    def gm_m3_s2(self, naif_id: int) -> float:
        """
        Return GM in SI units (m^3/s^2), using header constants when available.

        Supported:
        - Major-body barycenters: NAIF 1..9
        - Sun: NAIF 10
        - Earth-Moon barycenter: NAIF 3 (BODY399_GM + BODY301_GM when the
          file has no EMB key, as in EPM)
        - Moon (NAIF 301) derived from GM_EMB and EMRAT when present:
                GM_MOON = GM_EMB / (1 + EMRAT)

        Notes:
        - Many JPL DE and INPOP headers store GMs in AU^3/day^2.
        - Some store km^3/s^2. We detect the unit from Sun GM magnitude.
        """
        c = self.constants()
        nid = int(naif_id)
        gm_raw: float

        # ---- Special case: Moon or Earth GM ----
        if nid in (301, 399):
            # Prefer direct SPICE-style key (e.g. DE440 gm_de440.tpc)
            direct_key = f"BODY{nid}_GM"
            if direct_key in c:
                gm_raw = float(c[direct_key])
            elif "GM_EMB" in c and "EMRAT" in c:
                # INPOP-style: derive from Earth-Moon barycenter GM + mass ratio
                gm_emb = float(c["GM_EMB"])
                emrat  = float(c["EMRAT"])
                if nid == 301:  # Moon
                    gm_raw = gm_emb / (1.0 + emrat)
                else:           # 399, Earth
                    gm_raw = gm_emb * (emrat / (1.0 + emrat))
            else:
                raise KeyError(
                    f"GM requested for NAIF {nid} but neither {direct_key} nor "
                    f"GM_EMB+EMRAT found in ephemeris constants."
                )

        # ---- General case: major bodies and Sun ----
        else:
            # 1) INPOP-style keys
            inpop_map = {10: "GM_SUN", 1:  "GM_MER", 2:  "GM_VEN", 3:  "GM_EMB", 4:  "GM_MAR", 
                        5:  "GM_JUP", 6:  "GM_SAT", 7:  "GM_URA", 8:  "GM_NEP", 9:  "GM_PLU",}

            candidates: list[str] = []
            candidates_a: list[str] = []
            if nid in inpop_map:
                candidates.append(inpop_map[nid])

            # 2) DE/JPL-style keys
            if nid == 10:
                candidates += ["GMS", "BODY10_GM", "GM10"]
            elif nid == 3:
                candidates += ["GMB", "GM3", "BODY3_GM"]
            elif 1 <= nid <= 9:
                candidates += [f"GM{nid}", f"BODY{nid}_GM"]

            # Try candidates in order
            key = None
            for k in candidates:
                if k in c:
                    key = k
                    break

            # 3) Asteroid GM keys: INPOP MA-style (MA0001) or SPICE-style (BODY2000001_GM)
            if key is None:
                asteroid_index = self._asteroid_index_from_object_id(nid)
                if asteroid_index is not None:
                    # TODO: for KBOs / MBAs with catalogue numbers >= 10000 the
                    #   4- and 5-digit MA formats below may not match the actual
                    #   key stored in the ephemeris (INPOP uses MA0001..MA0343 for
                    #   DE430 perturbers; a future extended file could use MA10000+
                    #   or a completely different convention).  The bare
                    #   f"MA{asteroid_index}" fallback catches this, but it relies
                    #   on the key being stored without zero-padding, which is not
                    #   guaranteed.  Audit against real KBO/TNO ephemeris files
                    #   before enabling high-index asteroid support.
                    candidates_a = [
                        f"MA{asteroid_index:04d}", f"MA{asteroid_index:05d}", f"MA{asteroid_index}",
                        f"BODY{nid}_GM",  # SPICE/DE440 style: BODY2000001_GM
                    ]
                    for k in candidates_a:
                        if k in c:
                            key = k
                            break

            if key is not None:
                gm_raw = float(c[key])
            elif nid == 3 and "BODY399_GM" in c and "BODY301_GM" in c:
                # Only reached when no EMB key exists (INPOP and DE files have
                # one). EPM .tpc files give Earth and Moon separately; the
                # barycenter's GM is by definition the sum of the two.
                gm_raw = float(c["BODY399_GM"]) + float(c["BODY301_GM"])
            else:
                tried = candidates + candidates_a
                raise KeyError(f"No GM constant found for NAIF {nid}. Tried: {tried}")

        # ---- conversion block: native units -> SI ----
        unit = self.detect_gm_unit()  # decided from the Sun GM magnitude
        if unit == "AU3/DAY2":
            return gm_raw * AU3_PER_DAY2_TO_M3_PER_S2
        if unit == "KM3/S2":
            return gm_raw * KM3_PER_S2_TO_M3_PER_S2
        raise RuntimeError(f"Unexpected GM unit '{unit}'")
    
    # ---- function to query time offset between TCB and TCG etc. ----
    def time_offset_jd(self, kind: str, jd_int: float | int | np.ndarray, jd_frac: float | np.ndarray) -> np.ndarray:
        """
        Query time-ephemeris offsets from CALCEPH for split Julian-date input.
        """
        from calcephpy import Constants

        jd0, dt, was_scalar = self._as_jd_parts(jd_int, jd_frac)

        k = kind.strip().upper()
        if k == "TT-TDB":
            target = 16
        elif k == "TCG-TCB":
            target = 17
        else:
            raise ValueError(f"Unsupported kind '{kind}'. Use 'TT-TDB' or 'TCG-TCB'.")

        center = 0
        unit = Constants.UNIT_SEC
        out = self.eph.compute_unit(jd0, dt, int(target), int(center), unit)
        y = np.asarray(out[0], dtype=np.float64)
        return y[0] if was_scalar else y
    
    # ---- function to check which time offsets are supported for current eph file ----
    def supports_time_offsets(self) -> dict[str, bool]:
        out = {}
        jd_int = np.array([2451545], dtype=np.int64)
        jd_frac = np.array([0.0], dtype=np.float64)
        for k in ["TT-TDB", "TCG-TCB"]:
            try:
                _ = self.time_offset_jd(k, jd_int, jd_frac)
                out[k] = True
            except Exception:
                out[k] = False
        return out
    
    def fileversion(self) -> str | None:
        try:
            v = self.eph.getfileversion()
            return v.strip() if isinstance(v, str) else None
        except Exception:
            return None

    # ---- functions relating to asteroids ----
    # Two ID spaces are in play, both plain ints, so nothing but care keeps them
    # apart:
    #   "asteroid index"     e.g. 1 for Ceres — what MAxxxx header keys use
    #   "CALCEPH object ID"  = Constants.ASTEROID + index (2000000 + index),
    #                          matching NAIF SPK IDs for numbered asteroids
    # Converting between them is the job of the helpers below — asteroid_object_id
    # (index -> ID), _asteroid_index_from_object_id (ID -> index, None below the
    # base), asteroid_index_from_any (user input in either space -> index), and the
    # two list wrappers. When adding a path that touches asteroid IDs, know which
    # space you are in and route through these rather than hardcoding 2000000.
    @staticmethod
    def asteroid_object_id(asteroid_index: int) -> int:
        """
        Convert a CALCEPH asteroid index to a CALCEPH object ID.
        """
        from calcephpy import Constants
        base = int(Constants.ASTEROID)
        return base + int(asteroid_index)

    @staticmethod
    def _asteroid_index_from_object_id(object_id: int) -> int | None:
        """
        Reverse mapping of CALCEPH asteroid object ID -> asteroid index.
        """
        from calcephpy import Constants
        base = int(Constants.ASTEROID)

        obj = int(object_id)
        # Limitation: `>= base` is a pure arithmetic guard, not an identity check,
        # so any small body in that ID range (e.g. a comet) passes as an asteroid.
        # Untested for odd designations: header files carry KBO indices above the
        # real catalogue range (~800000) that no position file here exercises.
        if obj >= base:
            return obj - base
        return None

    @classmethod
    def asteroid_index_from_any(cls, value: int) -> int:
        """
        Normalise a user-supplied asteroid ID to an asteroid index: object IDs
        (>= base) are converted, bare indices pass through unchanged.

        Separate from _asteroid_index_from_object_id because that one returns
        None below the base — its callers rely on it to reject planets, while
        user input needs the pass-through.
        """
        idx = cls._asteroid_index_from_object_id(value)
        return int(value) if idx is None else idx

    @classmethod
    def asteroid_object_ids(cls, asteroid_indices: list[int]) -> list[int]:
        return [cls.asteroid_object_id(i) for i in asteroid_indices]

    @classmethod
    def asteroid_indices_from_object_ids(cls, object_ids: list[int]) -> list[int]:
        out: list[int] = []
        for oid in object_ids:
            idx = cls._asteroid_index_from_object_id(int(oid))
            if idx is not None:
                out.append(idx)
        return out

    @staticmethod
    def _ma_indices_from_keys(keys: Iterable[str]) -> list[int]:
        import re

        out = {
            int(m.group(1))
            for k in keys
            for m in [re.fullmatch(r"MA0*([0-9]+)", k)]
            if m is not None and int(m.group(1)) > 0
        }
        return sorted(out)

    def asteroid_object_ids_in_file(self) -> list[int]:
        """
        Return all asteroid object IDs present in the auxiliary asteroid file.
        If no asteroid file is open, returns [].
        """
        if self.asteroid_eph is None:
            return []

        try:
            get_record2 = getattr(self.asteroid_eph, "getpositionrecordindex2", None)
            n = int(self.asteroid_eph.getpositionrecordcount())
            ids: set[int] = set()
            for i in range(1, n + 1):
                rec = None
                if get_record2 is not None:
                    try:
                        rec = get_record2(i)
                    except Exception:
                        rec = None
                if rec is None:
                    rec = self.asteroid_eph.getpositionrecordindex(i)
                if not rec:
                    continue
                target = int(rec[0])
                if self._asteroid_index_from_object_id(target) is not None:
                    ids.add(target)
            return sorted(ids)
        except Exception:
            # Fallback for bindings/builds that do not expose position-record APIs:
            # infer asteroid IDs from MAxxxx constants if available.
            try:
                keys = self._constants_from_handle(self.asteroid_eph).keys()
            except Exception:
                keys = []
            return self.asteroid_object_ids(self._ma_indices_from_keys(keys))

    def supplement_gm_from_header(self, path: str | Path) -> tuple[int, int]:
        """
        Parse a JPL ASCII DE header file (e.g. header.440, header.441) and
        inject asteroid GM constants (MA-style keys) into the merged cache.
        User-supplied header values take priority: they override any same-key
        value already present from the main ephemeris file.

        Only MA0xxx-style keys are injected — planet GMs and other header
        constants are intentionally skipped to avoid mixing unit systems.
        JPL DE headers always store values in AU³/day²; if the existing
        ephemeris uses km³/s² (detected from the Sun GM magnitude), the
        incoming values are converted accordingly.

        The unit detection result is locked in before the cache is modified so
        that injecting a GMS key from the header cannot flip the detected unit.

        Returns (added, overridden_keys): count of newly inserted keys and list of overridden key names.
        """
        import re

        # Lock unit detection BEFORE modifying the cache.  If we added GMS from
        # the header first, detect_gm_unit() would find GMS=2.96e-4 < 1 and
        # wrongly return AU3/DAY2 even when the main ephemeris is in km³/s².
        existing_unit = self.detect_gm_unit()

        incoming = parse_de_header(path)
        existing = self.constants()

        added = 0
        overridden_keys: list[str] = []
        for k, v in incoming.items():
            # Only inject asteroid MA keys — skip planet GMs, AU, CLIGHT, etc.
            if not re.fullmatch(r"MA\d+", k):
                continue
            # Header files are always AU³/day²; convert if existing cache is km³/s².
            if existing_unit == "KM3/S2":
                v = v * AU3_PER_DAY2_TO_KM3_PER_S2
            if k in existing:
                overridden_keys.append(k)
            else:
                added += 1
            existing[k] = v
        return added, overridden_keys

    def sun_j2_constants(self) -> tuple[float, float]:
        """Return (J2_Sun, R_Sun_m) from ephemeris header constants.

        Reads J2SUN (dimensionless) and the Sun equatorial radius (RSUN for INPOP,
        ASUN for JPL DE; both stored in km).  Falls back to DE440 values if absent.
        """
        c = self.constants()
        j2 = float(c.get("J2SUN", 2.196139e-7))
        r_km = float(c.get("RSUN") or c.get("ASUN") or 6.96e5)
        return j2, r_km * 1e3  # convert km -> m

    def asteroid_indices_with_gm_in_main_file(self) -> list[int]:
        """
        Return asteroid indices that have GM constants available (merged cache,
        including any values injected via supplement_gm_from_header).
        Handles both INPOP MA-style keys (MA0001) and SPICE-style keys (BODY2000001_GM).
        """
        import re
        from calcephpy import Constants
        # Use the merged+supplemented cache, not just the raw main-file handle.
        keys = self.constants().keys()
        indices: set[int] = set(self._ma_indices_from_keys(keys))
        # SPICE/DE440 style: BODY2000001_GM, BODY2000002_GM, ...
        # Matching any digit run and filtering on the asteroid base (2000000, the
        # NAIF offset small-body IDs start at) also picks up bodies with catalogue
        # numbers >= 1000000, whose IDs land at >= 3000000 (e.g. BODY3000001_GM)
        # and so fall outside a 2xxxxxx-only pattern.
        base = int(Constants.ASTEROID)
        for k in keys:
            m = re.fullmatch(r"BODY(\d+)_GM", k)
            if m and int(m.group(1)) >= base:
                indices.add(int(m.group(1)) - base)
        return sorted(indices)


# -------------------------
# JPL ASCII DE header parser
# -------------------------

def parse_de_header(path: str | Path) -> dict[str, float]:
    """
    Parse a JPL ASCII DE ephemeris header file (e.g. header.440, header.441)
    and return a dict of all header constants (uppercased keys).

    The file format has two relevant groups:
      GROUP 1040  — N constant names, space-separated across multiple lines
      GROUP 1041  — N values in Fortran double notation (D exponent), same order

    Values are returned as Python float64.
    """
    import re

    text = Path(path).expanduser().resolve().read_text()

    def _extract_group(text: str, group_id: int) -> str:
        m = re.search(rf"GROUP\s+{group_id}\s+(.*?)(?=GROUP\s+\d+|\Z)", text, re.DOTALL)
        if m is None:
            raise ValueError(f"GROUP {group_id} not found in '{path}'.")
        return m.group(1)

    # GROUP 1040: first token is the count, remainder are names
    tokens_1040 = _extract_group(text, 1040).split()
    n = int(tokens_1040[0])
    names = [t.upper() for t in tokens_1040[1 : n + 1]]
    if len(names) != n:
        raise ValueError(f"Expected {n} names in GROUP 1040, found {len(names)}.")

    # GROUP 1041: first token is the count, remainder are Fortran doubles
    tokens_1041 = _extract_group(text, 1041).split()
    n2 = int(tokens_1041[0])
    if n2 != n:
        raise ValueError(f"GROUP 1040 has {n} names but GROUP 1041 has {n2} values.")
    values = [float(t.upper().replace("D", "E")) for t in tokens_1041[1 : n + 1]]

    return dict(zip(names, values))


# -------------------------
# SPICE text kernel (TPC) parser
# -------------------------

def parse_spice_tpc(path: str | Path) -> dict[str, float]:
    """
    Parse a SPICE text kernel (.tpc) and return a dict of scalar float constants
    (uppercased keys).

    Only assignments whose right-hand side is a single scalar value are included;
    vector/array values (e.g. BODY399_RADII) are silently skipped.

    Format example:
        \\begindata
           BODY10_GM  = (  132712440043.1652  )
           EMRAT      =  81.30056888602061
        \\begintext
    """
    import re

    text = Path(path).expanduser().resolve().read_text()

    # Collect only the \begindata blocks, ignoring \begintext commentary.
    data_blocks = re.findall(r'\\begindata(.*?)(?:\\begintext|\Z)', text, re.DOTALL | re.IGNORECASE)
    data_text = "\n".join(data_blocks)

    out: dict[str, float] = {}
    # Match:  KEY = scalar   or   KEY = ( scalar )
    pattern = re.compile(
        r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*\(\s*([+-]?\d[\d.eEdD+\-]*)\s*\)'
        r'|([A-Za-z][A-Za-z0-9_]*)\s*=\s*([+-]?\d[\d.eEdD+\-]*)'
    )
    for m in pattern.finditer(data_text):
        if m.group(1):
            key, raw = m.group(1), m.group(2)
        else:
            key, raw = m.group(3), m.group(4)
        try:
            out[key.upper()] = float(raw.upper().replace('D', 'E'))
        except ValueError:
            pass
    return out
