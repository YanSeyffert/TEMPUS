#!/usr/bin/env bash
# Generic TEMPUS example runner. Usage:
#   ./run_tempus.sh config_de440.txt
#
# Configs are simple KEY=value files (sourced as bash). Multi-file values
# (EPH_FILES, AST_GM_FILES) are space-separated strings; EXTRA_ARGS holds any
# additional raw `tempus integrate` flags (e.g. "--out-rate --sun-j2").
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:?usage: $0 <config-file>}"
source "$CONFIG"

# Unquoted $EPH_FILES / $AST_GM_FILES / $EXTRA_ARGS: word-splitting is
# intentional so a space-separated value becomes multiple CLI arguments
# (paths and flags contain no spaces).
cmd=(tempus integrate
    --eph-file $EPH_FILES
    --jd-start "$JD_START_INT" "$JD_START_FRAC"
    --jd-end   "$JD_END_INT"   "$JD_END_FRAC"
    --jd-step  "$JD_STEP"
    --target   "$TARGET"
    --center   "${CENTER:-SSB}"
    --ext      "${EXT:-auto}"
    --integrator "${INTEGRATOR:-am12}"
    --out      "$OUT_CSV"
)

if [[ -n "${AST_FILE:-}" ]];     then cmd+=(--asteroid-file "$AST_FILE"); fi
if [[ -n "${AST_GM_FILES:-}" ]]; then cmd+=(--asteroid-gm-header $AST_GM_FILES); fi
if [[ -n "${AST_IDS:-}" ]];      then cmd+=(--asteroid-ids "$AST_IDS"); fi
if [[ -n "${EXTRA_ARGS:-}" ]];   then cmd+=($EXTRA_ARGS); fi

echo "+ ${cmd[*]}"
"${cmd[@]}"
