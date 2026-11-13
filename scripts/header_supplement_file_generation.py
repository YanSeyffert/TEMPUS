# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
EPHEM_DIR = HERE.parent / "data" / "ephem"   # where header_mba.* live and are read from

# ----------------------------------------------------------------------
# Paste the original INPOP-style lines here
# ----------------------------------------------------------------------

# Expected format:
# raw_text = r"""
# GM_Ceres        =       +1.3863904478558460000E-13
# GM_Pallas       =       +2.9882165103302160000E-14
# GM_Vesta        =       +3.9310096581073580001E-14
# GM_AST(554)	=	+9.8655294326978140000E-17
# GM_AST(488)	=	+3.6459680269551619999E-16
# GM_AST(419)	=	+2.2735474822040490001E-16
# GM_AST(337)	=	+7.2719617012796850003E-17
# """

# Input (not shipped): the INPOP19a GM_... constant lines, saved next to this script.
raw_text = (HERE / "gm_inpop19a.dat").read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# Parse
# ----------------------------------------------------------------------
entries = []

for line in raw_text.strip().splitlines():
    line = line.strip()

    m = re.match(r"GM_(Ceres|Pallas|Vesta)\s*=\s*([+\-0-9.Ee]+)", line)
    if m:
        name = m.group(1)
        val = float(m.group(2))
        ids = {"Ceres": 1, "Pallas": 2, "Vesta": 4}
        asteroid_id = ids[name]
        entries.append((f"MA{asteroid_id}", val, name))
        continue

    m = re.match(r"GM_AST\((\d+)\)\s*=\s*([+\-0-9.Ee]+)", line)
    if m:
        asteroid_id = int(m.group(1))
        val = float(m.group(2))
        entries.append((f"MA{asteroid_id}", val, f"AST({asteroid_id})"))

# sort numerically by asteroid id
entries.sort(key=lambda x: int(x[0][2:]))

# ----------------------------------------------------------------------
# Write DE-header style file
# ----------------------------------------------------------------------
def _fmt_fortran(v):
    return f"{v:.16E}".replace("E", "D")

names = [e[0] for e in entries]
values = [e[1] for e in entries]
n = len(entries)

lines = [
    "KSIZE=  0  NCOEFF=  0",
    "",
    "GROUP   1040",
    "",
    f"   {n}",
]

for i in range(0, n, 6):
    lines.append("   " + "   ".join(f"{nm:<14}" for nm in names[i:i+6]))

lines += [
    "",
    "GROUP   1041",
    "",
    f"   {n}",
]

for i in range(0, n, 3):
    lines.append("   " + "   ".join(_fmt_fortran(v) for v in values[i:i+3]))

lines.append("")

out_path = EPHEM_DIR / "header_mba.19a"
out_path.write_text("\n".join(lines))

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

_AU_KM = 149597870.7
_DAY_S = 86400.0
_AU3D2_TO_KM3S2 = _AU_KM**3 / _DAY_S**2

print(f"Wrote {out_path} ({n} MBA GM entries)")

for key, gm, label in entries:
    print(
        f"{key:<8} {gm:.6e} AU3/d2   "
        f"= {gm * _AU3D2_TO_KM3S2:.6e} km3/s2   ({label})"
    )