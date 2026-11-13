#!/usr/bin/env python3
# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""Fetch the TU Dresden (Klioner) time-ephemeris Chebyshev files.

Source: https://gaia.geo.tu-dresden.de/TimeEphemerides/
Each file covers +-100 yr about J2000 and is derived from INPOP19a:
  - forward  `tcx19a_<TCX>_..._pos_TCB.asc`: TCX - TCB in seconds as a function of TCB;
  - inverse  `tcx19a_TCB_..._pos_<TCX>.asc`: the same offset as a function of TCX.
Klioner names Mars "TCMar"; it is saved as "TCM" (the name used by TEMPUS and the
paper), so no manual renaming is needed. Files go to `data/time_ephem/` of this
repository wherever the script is run from. Existing files are skipped, so this
is safe to re-run.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
import shutil
import sys

BASE = "https://gaia.geo.tu-dresden.de/TimeEphemerides"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "time_ephem"

# body label -> timescale tag, as saved locally
BODIES: dict[str, str] = {
    "Sun":     "TCSun",
    "Mercury": "TCMer",
    "Venus":   "TCVen",
    "Earth":   "TCG",
    "Moon":    "TCL",
    "Mars":    "TCM",
    "Jupiter": "TCJup",
    "Saturn":  "TCSat",
    "Uranus":  "TCUra",
    "Neptune": "TCNep",
    "Pluto":   "TCPlu",
}

# local tag -> Klioner's tag, where they differ
REMOTE_TAG: dict[str, str] = {"TCM": "TCMar"}

# inverse products used by the notebooks (Earth and Mars, notebook 08)
INVERSE: list[str] = ["TCG", "TCM"]


def asc_name(tag: str) -> str:
    """Forward file name: TCX - TCB as a function of TCB."""
    return f"tcx19a_{tag}_m100_p100_asc_pos_TCB.asc"


def inverse_asc_name(tag: str) -> str:
    """Inverse file name: TCX - TCB as a function of TCX."""
    return f"tcx19a_TCB_m100_p100_asc_pos_{tag}.asc"


def _download(url: str, path: Path, timeout: float, attempts: int = 2) -> bool:
    """Download url to path unless it exists; True if the file is there afterwards."""
    if path.is_file():
        print(f"  have    {path.name} ({path.stat().st_size/1e6:.1f} MB)", flush=True)
        return True
    tmp = path.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=timeout) as r, tmp.open("wb") as fh:
                shutil.copyfileobj(r, fh)
            tmp.replace(path)     # atomic: a partial download never looks complete
            print(f"  fetched {path.name} ({path.stat().st_size/1e6:.1f} MB)", flush=True)
            return True
        except Exception as e:
            tmp.unlink(missing_ok=True)
            again = " — retrying" if attempt < attempts else ""
            print(f"  FAILED  {path.name}: {e}{again}", file=sys.stderr, flush=True)
    return False


def fetch_klioner_asc(dest_dir: str | Path = DEFAULT_DEST,
                      tags: list[str] | None = None,
                      inverse_tags: list[str] | None = None,
                      timeout: float = 120.0) -> dict[str, Path]:
    """Download the forward and inverse files, returning {file name: path} of
    those present afterwards; failures are reported on stderr."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for tag in (tags if tags is not None else list(BODIES.values())):
        path = dest / asc_name(tag)
        if _download(f"{BASE}/TCXminusTCB/{asc_name(REMOTE_TAG.get(tag, tag))}", path, timeout):
            out[path.name] = path

    for tag in (inverse_tags if inverse_tags is not None else INVERSE):
        path = dest / inverse_asc_name(tag)
        if _download(f"{BASE}/TCBminusTCX/{inverse_asc_name(REMOTE_TAG.get(tag, tag))}", path, timeout):
            out[path.name] = path

    return out


if __name__ == "__main__":
    wanted = len(BODIES) + len(INVERSE)
    got = fetch_klioner_asc(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST)
    if len(got) < wanted:
        sys.exit(f"{wanted - len(got)} file(s) failed — re-run to retry; files already present are skipped.")
