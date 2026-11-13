# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

from pathlib import Path
import re

_NAMED_IDS = {1: "Ceres", 2: "Pallas", 4: "Vesta"}
_AU_KM = 149597870.7
_DAY_S = 86400.0
_AU3D2_TO_KM3S2 = _AU_KM**3 / _DAY_S**2

# Expected format:
#      1	+1.396435324290342781921E-13
#      2	+3.164549125428774180438E-14
#      4	+3.854808120050912648861E-14
#      7	+1.925754939766526046426E-15
#    324	+1.571856877286163904832E-15
#     12	+3.774047812146744695750E-16
#     25	+1.655418753805156676132E-16
#     27	+3.375241515631743300544E-16
#     28	+2.884901550787565217645E-16
#     44	+4.039038343331418109240E-17
#     46	+7.794747088257999808645E-17
#     94	+7.736131611859339286650E-16
#    192	+2.404944225008402665445E-16
#    200	+3.022304221751691725093E-16
#    201	+1.899912250682851638401E-16
#    416	+1.097554587283005189628E-16
#    516	+7.928720902783384000862E-17
#    694	+1.361995832460915792819E-16
#  50000	+2.132289999999999999962E-13


def _fmt_fortran(v):
    return f"{v:.16E}".replace("E", "D")


def generate_header_supplement(input_path: Path | str, output_name: str = "header_mba.dat") -> Path:
    """Parse a two-column asteroid GM file and write a DE-header supplement.

    Input format per line:  <integer_id>   <GM_value_AU3/d2>
    """
    input_path = Path(input_path)
    raw_text = input_path.read_text(encoding="utf-8")

    entries = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\s+([+\-0-9.Ee]+)", line)
        if m:
            asteroid_id = int(m.group(1))
            if asteroid_id >= 20_000_000:
                asteroid_id -= 20_000_000
            val = float(m.group(2))
            label = _NAMED_IDS.get(asteroid_id, f"AST({asteroid_id})")
            entries.append((f"MA{asteroid_id}", val, label))

    entries.sort(key=lambda x: int(x[0][2:]))

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

    lines += ["", "GROUP   1041", "", f"   {n}"]
    for i in range(0, n, 3):
        lines.append("   " + "   ".join(_fmt_fortran(v) for v in values[i:i+3]))
    lines.append("")

    out_path = input_path.parent / output_name
    out_path.write_text("\n".join(lines))

    print(f"Wrote {out_path} ({n} MBA GM entries)")
    for key, gm, label in entries:
        print(
            f"{key:<8} {gm:.6e} AU3/d2   "
            f"= {gm * _AU3D2_TO_KM3S2:.6e} km3/s2   ({label})"
        )

    return out_path


if __name__ == "__main__":
    # Usage: python header_supplement_from_twocol.py <two-column GM table> [output name]
    # The header is written next to the input table.
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: header_supplement_from_twocol.py <input table> [output name]")
    generate_header_supplement(Path(sys.argv[1]), *sys.argv[2:3])
