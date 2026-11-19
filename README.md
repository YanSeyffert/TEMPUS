# TEMPUS Integrator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24199182.svg)](https://doi.org/10.5281/zenodo.24199182) [![arXiv](https://img.shields.io/badge/arXiv-2609.08979-b31b1b.svg)](https://arxiv.org/abs/2609.08979)

TEMPUS is a self developed lightweight tool for **ephemeris-driven time-coordinate integration**. *Tempus* is the latin word for *time*. 

It computes relativistic time-coordinate offsets (e.g., **TCM–TCB**) by numerically integrating post-Newtonian models along a target body's ephemeris-defined trajectory/worldline. TEMPUS is *ephemeris-agnostic* and supports JPL/INPOP/EPM solutions via **CALCEPH** (native binary ephemerides such as `.jpl`, `.tcheb`, `.dat`), which can also handle Spice Kernels (eg. `de440.bsp`, plus time and constants kernels).

Enables the following studies:
- Compute a planetocentric time-coordinate (TCX) at the body center from barycentric coordinate time (TCB)
- Compare the resulting TCX–TCB model across ephemeris releases and enables forming weighted time-scales (as the ILuRS product does with the Lunar timescale TCL)
- Provide time-scale transformations supporting planetary reference frames and PNT architecture studies

For usage of TEMPUS, see section Quickstart below and `examples/`; for concepts and detailed walkthroughs, see the numbered notebooks under `notebooks/`. The model and its validation are described in the TEMPUS paper ([arXiv:2609.08979](https://arxiv.org/abs/2609.08979)).


## Repository layout

Core library code lives in `src/tempus/`:

- **`ephem.py`** Ephemeris backend (`CalcephEphem`): reads INPOP, DE and EPM files and SPICE kernels through CALCEPH, and returns states (`state_jd`, on split Julian dates) and GM values in **SI units** (m, m/s, m³/s²).

- **`bodies.py`** Body names and their NAIF IDs, and the default perturber lists.

- **`integrate.py`** Numerical integrators and their runtime selection: trapezoidal, RK4 and Adams–Moulton of several orders (default `am12`).

- **`model.py`** Post-Newtonian time-coordinate model and construction of the integrand, calling the selected integrator; also the split-JD time types and the exact time-scale relations.

- **`output.py`** Output columns, the CSV writer and reader, and the JSON metadata sidecar.

- **`chebycode/`** Piecewise-Chebyshev fitting and evaluation behind `tempus chebyshev`.

- **`cli.py`** The `tempus` command line, with the subcommands `integrate` and `chebyshev`.

Examples and CLI entry points are thin wrappers around the library code. Large binary inputs are stored locally under `data/` and are not committed.


## Model (planet center)

At the target body center $X$ (e.g. Mars barycenter), TEMPUS numerically integrates the post-Newtonian rate $d\mathrm{TCX}/d\mathrm{TCB} - 1$ to obtain $\Delta(t) \equiv \mathrm{TCB}(t) - \mathrm{TCX}(t)$:

```math
\frac{d\mathrm{TCX}}{d\mathrm{TCB}} - 1 \;=\; \frac{\alpha_X}{c^2} + \frac{\beta_X}{c^4}
```

**O(c⁻²) term**

```math
\alpha_X = -\left(\frac{1}{2} v_X^2 + U + \Delta U_{J_2,\odot}\right), \qquad U = \sum_{A \neq X} \frac{GM_A}{r_{AX}}
```

where $\Delta U_{J_2,\odot}$ is an optional solar-oblateness (quadrupole) contribution, off by default and enabled with `--sun-j2` — see the TEMPUS paper and `notebooks/05_sensitivity_study.ipynb`.

**O(c⁻⁴) term**

```math
\beta_X = -\frac{1}{8} v_X^4 + \frac{1}{2} U^2 + \sum_{A \neq X} \frac{GM_A}{r_{AX}} \left[4\,(\mathbf{v}_X \cdot \mathbf{v}_A)-\frac{3}{2} v_X^2-2\, v_A^2+\frac{1}{2} (\mathbf{a}_A \cdot \mathbf{r}_{AX})+\frac{1}{2} \left(\frac{\mathbf{v}_A \cdot \mathbf{r}_{AX}}{r_{AX}}\right)^2+\sum_{B\ne A} \frac{GM_B}{r_{BA}}\right]
```

where:
- $\mathbf{v}_X$ is the barycentric velocity of the target body center (from the ephemeris),
- $U$ is the Newtonian gravitational potential at the target center due to all other bodies,
- $GM_A$, $\mathbf{v}_A$, $\mathbf{a}_A$ are the gravitational parameter, barycentric velocity, and barycentric acceleration of perturbing body $A$,
- $r_{AX} = \lVert\mathbf{r}_{AX}\rVert$ is the distance between body $A$ and the target,
- $c$ is the speed of light.

## Quickstart

### Installation
From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs the core runtime dependencies: `numpy`, `scipy`, `matplotlib`, `calcephpy`.

If the installation or the data download fails on an older Linux system (e.g. CentOS 7), see [docs/INSTALL_OLDER_LINUX.md](docs/INSTALL_OLDER_LINUX.md).

To also run the notebooks under `notebooks/`, install the additional packages they use beyond the core library (`astropy` for time-scale diagnostics, `spiceypy` for SPICE-based cross-checks, plus Jupyter itself):
```bash
pip install jupyter ipykernel astropy spiceypy 
```

### Ephemeris and kernel data

Large data files are stored locally as:
- `data/ephem/` - CALCEPH ephemerides ( `.dat`, `.jpl`, `.tcheb`)
- `data/kernels/` - SPICE kernels (`.bsp`, `.tpc`, `.tls`)
- `data/time_ephem/` - TCX−TCB time ephemerides (`.asc`) from Klioner and INPOP, used by the validation notebooks

You can quickly download the ephemerides with the provided script:

```
bash scripts/download_data.sh inpop de440 de430 asteroids epm
```

The script's additional group `inpop_asc` adds INPOP's own TCG−TCB file, and `scripts/fetch_klioner_asc.py` downloads Klioner's time ephemerides; both are only needed for the validation notebooks. Only the locally built GM header supplements (`header_mba.19a`, `header_mba.21a`, `header_kbo.440`) are included in the repository. For more info see [scripts/README.md](scripts/README.md).

### Examples
TEMPUS provides a CLI for reproducible/automatable runs. See the folder `examples/` for ready-to-run configs (for each ephemeris family: DE, INPOP, EPM) driven by `run_tempus.sh`. For more infos see [examples/README.md](examples/README.md). The `notebooks/` provide the full walkthroughs; special attention to the end of notebook 00 and notebooks 07-09, if you want to skip reading the detailed validation notebooks/tests.

Quick CLI example computing TCB-TCM for Mars (w.r.t. SSB) over 100 years, which outputs a .csv file and a corresponding .json file with metadata of the run/command:

```bash
# Compute TCB-TCM for Mars with CALCEPH backend, 1977 to ~2077 in JD(TCB), zeroed at the 1977 origin
tempus integrate \
  --eph-file data/ephem/inpop19a_TCB_m100_p100_tcg.dat \
  --jd-start 2443144 0.5 \
  --jd-end   2479669 0.5 \
  --jd-step  0.75 \
  --target MARS_BARYCENTER \
  --out output/mars_1977_2077_tcm.csv \
  --zero
```

Notes:
- `--jd-start` is 1 January 1977, 0h TCB. The run doesn't have to start exactly at the 1977 origin (1977-01-01 00:00:00 TAI, where TCG-TCB = 0 by IAU convention): `--zero` puts TCB-TCM = 0 there, at JD_TCB 2443144 0.5003725 on this TCB axis (JD_TDB 2443144 0.5003724992418982 for TDB ephemerides) — see `notebooks/00_tempus_background.ipynb`. 
- `--jd-end` is 100 Julian years (36525 days) later.
- `--jd-step 0.75` (18 h) is a standard ephemeris cadence and plays well with the default `am12` integrator.
- Time sampling is controlled by these three arguments above. **All JD inputs are interpreted in the ephemeris's native timescale**; TDB for most ephemerides, TCB for the TCB releases of INPOP. For TCB-scale ephemerides, `--input-jd-are-tdb` lets you give TDB inputs instead: the grid is built in TDB and every point is internally converted to TCB (via IAU 2006 Res. B3, linear relation). Output CSV includes split time columns `JD_int`, `JD_frac`, and `SecondsPastJ2000` (in the native timescale of the ephemeris).
- `--target MARS_BARYCENTER` (NAIF ID 4) selects the TCX body — here TCX = TCM. 
- `--center` when omitted defaults to the solar system barycenter (thus TCB) as reference. 
- `--ext` when omitted defaults to resolving the perturbing bodies automatically (Sun + planets + Pluto, excluding the target; + given asteroids); an explicit comma-separated list is also accepted.
- `--integrator` defaults to `am12` (Adams-Moulton 12-step). Other options: `trapz`, `rk4`, `am4`, `am8`, `am10`, `amk`.
- `--zero` writes the offset already zeroed at the 1977 origin, or at a given `--zero INT FRAC`; without it, the timescale offset is 0s at the first sample.
- Omitting `--out` auto-generates the filename as `{tcx}_{mYYYY}_{pYYYY}_{ephem}.csv` — target timescale tag, whole years before/after J2000 covered by the grid, ephemeris stem (the run above would yield `tcmars_m0023_p0077_inpop19a_tcb.csv`).
- More options available via `tempus integrate --help`.

### Chebyshev products: `tempus chebyshev`

Fit a piecewise Chebyshev polynomial series to an integrated CSV and write an INPOP-style `.asc` file:

```bash
tempus chebyshev output/mars_1977_2077_tcm.csv --max-err 1e-9
```

- Defaults (`--order 11 --granule-len 4 --npoints 10 --dist 2`) are tuned for timescale-difference series; `--column` selects another CSV column.
- `--use-rate` feeds the rate column (from `integrate --out-rate`) into the fit as derivative information and enables `--dist 1` (C¹ node distribution). Measured guidance: derivative information only helps for sparse sampling (≲ 8 samples per granule — there it wins by orders of magnitude); at the INPOP-standard 0.75 d cadence with 3–4 d granules it changes nothing, and `--dist 1` can be slightly *worse* than the default `--dist 2`.
- `--also-wrt-tcx` additionally re-labels the offset onto the local timescale TCX and writes a **second** file (`out.asc` native, `out_tcx.asc` wrt TCX) — they are different functions (~0.7 µs/century for TCG, ~0.3 for TCM; Klioner 2026), not a constant offset apart. TCX is never written alone, so a `_tcx.asc` file can never be mistaken for a normal product; it always shares its native sibling's run. Theory, demonstration, and model-independent validation against Klioner's own inverse products for Earth and Mars in `notebooks/08_argument_timescale.ipynb`.
- `--anchor` zeroes the fitted series at an epoch other than the integration start, for CSVs written without `integrate --zero`: bare `--anchor` uses the 1977 origin (where all TCX defined by the IAU so far coincide by definition), or `--anchor INT FRAC` a custom split-JD epoch. This is a standalone output convention; with `--also-wrt-tcx` it also fixes the zero convention of the TCX argument labels (without it, the relabelled axis is offset from the conventional TCX by the series' value at the integration start). Every `.asc` records its argument timescale and zero convention in its title line, so which convention a file uses is never ambiguous from the outside.
- A **fit-quality gate** runs by default and the command exits non-zero on failure, so scripts cannot silently ship a bad `.asc`. With `--max-err` the maximum residual against the samples is the binding criterion; without it the coefficient-tail convergence is checked (note: near machine-precision fits the tail flattens into the float64 noise floor and reports `converged: False` — prefer an explicit `--max-err` budget). `--auto` retries with higher order / smaller granules; `--no-validate` skips the gates.
- The `.asc` title line carries provenance (column, target, ephemeris, timescale). Load it back with `tempus.chebycode.ChebyshevSeries.from_asc()`.

### Differences of local timescales (non-SSB center)

```bash
tempus integrate \
  --eph-file data/ephem/inpop19a_TCB_m100_p100_tcg.dat \
  --jd-start 2443144 0.5 \
  --jd-end   2479669 0.5 \
  --jd-step  0.75 \
  --target MOON \
  --center EARTH_CENTER \
  --zero 
```

A non-SSB `--center` computes `TC_center − TC_target` (here TCG−TCL) **as a function of TCB** by BCRS subtraction of two SSB-centered integrations (`model.Delta_TCX_TCY`). The output column is named accordingly (`TCG_minus_TCL`) and the run metadata records `"difference_mode": "bcrs_subtraction", "simultaneity": "TCB"`. The Lorentz-boost / simultaneity term −**v**_E·**r**_LE/c² (≲136 µs monthly oscillation for Earth–Moon) is deliberately **not** applied, which would convert to the geocentric-simultaneity description. See `notebooks/07_example_tcl_tcg_subtraction.ipynb` for both descriptions side by side.

#### Asteroid GM priority

When running with asteroid perturbations (which is recommended for consistency with the loaded ephemeris, see `/examples` folder), the asteroid gravitational parameters (GM) can come from two sources:

1. **Main ephemeris file** — GM constants embedded in the `.dat` / `.tcheb` file (e.g. INPOP MA-style keys).
2. **User-supplied header file** (`--asteroid-gm-header`) — a JPL ASCII DE header (e.g. `header.440`) carrying MA-style GM constants.

**User-supplied values always take priority**: if the same asteroid key appears in both sources, the value from `--asteroid-gm-header` overwrites the main-file value. This allows substituting custom or updated mass estimates without modifying the ephemeris file itself. The CLI prints a summary of new and overridden keys for each header file loaded.


## Developer notes

See [Developer_Notes.md](docs/DEVELOPER_NOTES.md)

## Contact / Authors

Maintained by: Yan SEYFFERT, Agnès FIENGA.

## Citation
If you use TEMPUS, please cite the TEMPUS paper ([arXiv:2609.08979](https://arxiv.org/abs/2609.08979)). The archived release can be cited in addition: [10.5281/zenodo.24199182](https://doi.org/10.5281/zenodo.24199182). Citation metadata is in the [`CITATION.cff`](CITATION.cff) (also available via GitHub's "Cite this repository" button).

## License

TEMPUS is released under the **BSD 3-Clause License** (BSD-3-Clause) — see [LICENCE](LICENCE).

Copyright (c) 2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD). Authors: Yan SEYFFERT, Agnès FIENGA.
