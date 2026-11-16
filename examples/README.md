# examples/

Runnable CLI examples: one generic runner plus one config per ephemeris release. Data files are expected under `../data/` — fetch them with `../scripts/download_data.sh` (background on the ephemeris files and manual URLs in `notebooks/00_tempus_background.ipynb` §2.2–2.3).

```bash
./run_tempus.sh config_de440.txt      # JPL DE440    (TDB)
./run_tempus.sh config_de430.txt      # JPL DE430    (TDB)
./run_tempus.sh config_inpop21a.txt   # INPOP21a     (TDB; TCB flavour commented)
./run_tempus.sh config_inpop19a.txt   # INPOP19a     (TDB; TCB flavour commented)
./run_tempus.sh config_epm2021.txt    # EPM2021      (TDB, SPICE format)
```

All configs target the **Moon → TCL-TCB** so the outputs are directly comparable across ephemerides. The INPOP configs default to the TDB flavour for the same reason — DE and EPM only exist in TDB; the TCB-native files are one commented line away.

Run times on a laptop (Intel Core Ultra 9 185H, WSL2; each run uses one core). The number of small-body perturbers dominates:

| Config | Small bodies | Span | Run time |
|---|---|---|---|
| `config_de440.txt` | 5 (Ceres, Pallas, Vesta, Iris, Bamberga) | 200 yr | 3 s |
| `config_epm2021.txt` | 244 | 82 yr | 1.2 min |
| `config_de430.txt` | 343 | 82 yr | 2.1 min |
| `config_inpop19a.txt` | 353 | 200 yr | 5.7 min |
| `config_inpop21a.txt` | 353 | 200 yr | 5.6 min |

## Config format

A config is a plain `KEY=value` file (sourced as bash):

- multi-file values are space-separated strings
  (`EPH_FILES="de440.bsp gm_de440.tpc"`)
- `EXTRA_ARGS` holds any additional flags, written exactly as you'd pass them
  on the command line (`EXTRA_ARGS="--out-rate --out-tt-tdb"`)
- `CENTER`, `EXT`, `INTEGRATOR` may be omitted — they default to
  `SSB` / `auto` / `am12`

Copy a config and edit to build your own run. The runner prints the fully assembled `tempus integrate` command before executing it.

## Current option notes

- **`--zero`** (in every config's `EXTRA_ARGS`) writes TCB−TCX already zeroed
  at the IAU 1977 origin, which each config's span contains; without it, the
  offset is 0 at the first sample, and wouldn't be compliant with IAU def. yet.
- All O(c⁻⁴) terms are always included unless you add `--pn1` to `EXTRA_ARGS` 
which limits to O(c⁻²)
- **All JD inputs are in the ephemeris-native timescale** (TDB for DE/EPM and
  the INPOP TDB flavours; TCB for the INPOP TCB flavours). For TCB files,
  `--input-jd-are-tdb` accepts TDB inputs and converts the whole grid
  (start/end/step) to TCB exactly.
- On the plain CLI, omitting `--out` auto-generates the filename as
  `{tcx}_{mYYYY}_{pYYYY}_{ephem}.csv`; the runner script always sets `OUT_CSV`
  explicitly.
- **`tempus chebyshev out.csv --use-rate --max-err 1e-9`** fits the CSV
  into an INPOP-style `.asc` Chebyshev file, with fit-quality gates and a
  non-zero exit code on failure — see the main README for the details.
- **Non-SSB centers**: `--target MOON --center EARTH_CENTER` outputs
  `TCG_minus_TCL` wrt TCB via BCRS subtraction of two SSB-centered runs
  (`notebooks/07_example_tcl_tcg_subtraction.ipynb`).
- `--sun-j2` adds the solar quadrupole contribution.
- `--out-tt-tdb` / `--out-tcg-tcb` read the file's own time ephemeris — INPOP
  releases only (TDB and TCB flavour respectively); each INPOP config's
  `EXTRA_ARGS` comment says which to swap in if you switch flavours.
- Every run writes a `.json` metadata sidecar next to the output CSV.

For the full argument walkthrough see `notebooks/00_tempus_background.ipynb` §7; the validation/comparison workflows live in the numbered `0X_*` notebooks.
