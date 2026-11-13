#!/usr/bin/env bash
# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# Download the public ephemeris/kernel files used by examples/*.txt into the
# correct data/ subfolders. Safe to re-run — existing files are skipped.
#
# Usage:
#   ./download_data.sh                    # everything (~2.5 GB total)
#   ./download_data.sh inpop de440 lsk    # only the named groups
#
# Groups: inpop de440 de430 asteroids epm lsk inpop_asc
#   inpop_asc: INPOP19a's own TCG-TCB as an ASCII Chebyshev file, used by the
#   validation notebooks (data/time_ephem/). Klioner's time ephemerides for the
#   same notebooks come from scripts/fetch_klioner_asc.py.
#
# NOT covered here (see notebooks/00_tempus_background.ipynb §2.3):
#   - header_mba.19a, header_mba.21a, header_kbo.440 — generated locally by
#     scripts/header_supplement_*.py, already committed to the repo.
#   - LTE440 (notebook 04): lte440.bsp and lte440.tpc from
#     https://github.com/xlucn/LTE440/releases, into data/LTE440-tcl_tdb/.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (this script lives in scripts/)

EPHEM_DIR=data/ephem
KERNEL_DIR=data/kernels
TIME_DIR=data/time_ephem
mkdir -p "$EPHEM_DIR" "$KERNEL_DIR" "$TIME_DIR"

SEL_GROUPS=("$@")
want() {
    [[ ${#SEL_GROUPS[@]} -eq 0 ]] && return 0   # no args -> everything
    for g in "${SEL_GROUPS[@]}"; do [[ "$g" == "$1" ]] && return 0; done
    return 1
}

FAILED=()
# --show-progress needs wget >= 1.16; older ones (e.g. CentOS 7's 1.14) get a plain progress bar
if wget --help 2>&1 | grep -q -- '--show-progress'; then
    WGET_OPTS=(-q --show-progress)
else
    WGET_OPTS=(--progress=bar:force)
fi
fetch() {  # fetch <dest-dir> <url> [dest-filename]
    local dir="$1" url="$2"
    local name="${3:-$(basename "$url")}"
    if [[ -f "$dir/$name" ]]; then
        echo "  skip (exists): $dir/$name"
    else
        echo "  fetching: $name"
        # download under a .part name, so an interrupted run never leaves a
        # truncated file that the next run would skip as "exists"; a failed
        # download is reported and the remaining files are still fetched
        if wget "${WGET_OPTS[@]}" -O "$dir/$name.part" "$url"; then
            mv "$dir/$name.part" "$dir/$name"
        else
            rm -f "$dir/$name.part"
            echo "  FAILED:   $name  <- $url" >&2
            FAILED+=("$name")
        fi
    fi
}

if want inpop; then
    echo "== INPOP19a/21a (IMCCE FTP) =="
    fetch "$EPHEM_DIR" https://ftp.imcce.fr/pub/ephem/planets/inpop19a/inpop19a_TCB_m100_p100_tcg.dat
    fetch "$EPHEM_DIR" https://ftp.imcce.fr/pub/ephem/planets/inpop19a/inpop19a_TDB_m100_p100_tt.dat
    fetch "$EPHEM_DIR" https://ftp.imcce.fr/pub/ephem/planets/inpop21a/inpop21a_TCB_m100_p100_tcg.dat
    fetch "$EPHEM_DIR" https://ftp.imcce.fr/pub/ephem/planets/inpop21a/inpop21a_TDB_m100_p100_tt.dat
fi

if want de440; then
    echo "== JPL DE440 =="
    fetch "$KERNEL_DIR" https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp
    fetch "$KERNEL_DIR" https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
    fetch "$KERNEL_DIR" https://ssd.jpl.nasa.gov/ftp/eph/planets/ascii/de440/header.440
    # short variant (~30 MB, planets only, no libration/nutation): uncomment if wanted
    #fetch "$KERNEL_DIR" https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp
fi

if want de430; then
    echo "== JPL DE430 =="
    fetch "$KERNEL_DIR" https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp
    fetch "$KERNEL_DIR" https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de431.tpc
    fetch "$KERNEL_DIR" https://ssd.jpl.nasa.gov/ftp/eph/planets/ascii/de430/header.430_572
fi

if want asteroids; then
    echo "== Asteroid/TNO trajectories =="
    fetch "$KERNEL_DIR" https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n373s.bsp
    fetch "$KERNEL_DIR" https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de430/ast343de430.bsp
fi

if want epm; then
    echo "== EPM2021 (IAA RAS) =="
    fetch "$KERNEL_DIR" https://ftp.iaaras.ru/pub/epm/EPM2021/SPICE/epm2021.bsp
    fetch "$KERNEL_DIR" https://ftp.iaaras.ru/pub/epm/EPM2021/SPICE/epm2021.tpc
    fetch "$KERNEL_DIR" https://ftp.iaaras.ru/pub/epm/EPM2021H/DE/header.21h
fi

if want lsk; then
    echo "== Leap-second kernel =="
    fetch "$KERNEL_DIR" https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
fi

if want inpop_asc; then
    echo "== INPOP19a TCG-TCB, ASCII Chebyshev (IMCCE FTP) =="
    # IMCCE ships it inside the full ASCII archive (67 MB); only this file is kept.
    name=inpop19a_TCB_m100_p100_asc_pos_TCG.asc
    if [[ -f "$TIME_DIR/$name" ]]; then
        echo "  skip (exists): $TIME_DIR/$name"
    else
        echo "  fetching: inpop19a_TCB_m100_p100_asc.tar.gz (keeping $name)"
        part="$TIME_DIR/inpop19a_TCB_m100_p100_asc.tar.gz.part"
        if wget "${WGET_OPTS[@]}" -O "$part" https://ftp.imcce.fr/pub/ephem/planets/inpop19a/inpop19a_TCB_m100_p100_asc.tar.gz \
           && tar xzf "$part" -C "$TIME_DIR" "$name"; then
            :
        else
            echo "  FAILED:   $name" >&2
            FAILED+=("$name")
        fi
        rm -f "$part"
    fi
fi

if (( ${#FAILED[@]} )); then
    echo "Failed: ${FAILED[*]} — re-run to retry; files already present are skipped." >&2
    exit 1
fi

echo "Done."
