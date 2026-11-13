# Developer Notes (TEMPUS)

Why TEMPUS is structured the way it is, and how to extend it. This document deliberately does not restate current API signatures, parameter lists, or CLI flags — those live in the code, `--help`, and module docstrings.

For usage, see `README.md` and `examples/`; for concepts and walkthroughs, the numbered notebooks under `notebooks/` (start with `00_tempus_background.ipynb`).

## Module boundaries

- `ephem.py` — ephemeris I/O only. Reads ephemeris files, returns state vectors and GM values in SI units. Knows nothing about the physics model.
- `integrate.py` — numerical integration only. Every integrator has the same  signature (`integrate(y, t) -> I` with `I[0] = 0`) so the physics model can   select one at runtime without caring how it works internally.
- `model.py` — the physics. Builds the post-Newtonian integrand from ephemeris states and calls the selected integrator. This is the only file that should contain physics; `ephem.py` and `integrate.py` should never need to know what a PN term is. It also holds the split-JD time types and the exact, ephemeris-free time relations (TDB↔TCB, the relabelling onto TCX).
- `output.py` — output column assembly (CSV + JSON metadata sidecar), and reading a run back (`read_csv`) for `tempus chebyshev`.
- `chebycode/` — piecewise-Chebyshev fitting and evaluation, decoupled from the integration pipeline; `tempus chebyshev` is its CLI front end. General-purpose (it fits orbits and orientation angles as well as time offsets), so it knows nothing about time scales. See its module docstring for the API.
- `bodies.py` — NAIF ID / name parsing (`BodyRef`), default perturber lists.
- `cli.py` — a thin, reproducible runner with two subcommands, `integrate` and `chebyshev`: parse args, call the library, write outputs. Physics never goes here.


## Ephemeris backends: one today, door open for more

`CalcephEphem` is deliberately the **single** backend. A `SpiceEphem` (SpiceyPy) backend existed early on and was removed once CALCEPH proved it reads SPICE SPK/PCK files directly — one backend to maintain instead of two covering the same files.

The door stays open for re-adding SPICE (or another library) as a backend. A new backend must provide what `model.py` and `cli.py` actually consume. The essentials are:

- `state_jd(jd_int, jd_frac, target, center)` → state in SI (m, m/s),  split-JD input in the file's native timescale
- `gm_m3_s2(naif_id)` → GM in SI, whatever the file's storage unit
- `timescale()` / `timescale_enum()` → TDB vs TCB detection (drives the L_B rescaling in the model)
- `constants()`, `time_offset_jd(kind, ...)`, `fileversion()`, `close()`
- the asteroid helpers, if the backend should support `--asteroid-file`

Keep unit and timescale conversions **inside** the backend; the model layer must never see backend-native units or single-float JDs.

## Why split-JD instead of plain float JD

A single `float64` JD (~2.45×10⁶) only has microsecond-level resolution because the large integer part eats most of the mantissa. TEMPUS represents every epoch as `(jd_int, jd_frac)` instead, and the invariant that matters is: **never recombine int+frac into a single float internally** — only at the boundary (file I/O, display). `TimeScalar`/`jd_split`/`jd_grid` in `model.py` exist to make this invariant easy to keep; see `notebooks/00_tempus_background.ipynb` §1 and §6 for the worked-out numbers.

## Chebyshev products: a separate step

The sampled CSV is the primary product. `tempus chebyshev` turns a finished run into an `.asc` file as a **separate subcommand** that reads the CSV and its JSON sidecar, rather than as an output option of `integrate`. One integration can therefore be refitted with other settings without re-integrating, and the fit stays decoupled from the physics.

Because `chebycode` is compatible with orbit and orientation fitting (notebook 06), a change to its defaults must be tested on curved data too, not only on TEMPUS's near-linear time offsets — aligning granule edges to whole JD failed exactly that test (see `TODO.md`).

## Contribution workflows

### Adding a new integrator
1. Implement it in `integrate.py` with the standard signature.
2. Register it in `get_integrator(name)`.
3. Add a regression test against an analytic function (see `tests/test_integrate_am10.py` for the pattern: integrate a known function on a uniform grid, assert against a fixed expected array).
4. List it in the `--integrator` help text in `cli.py` (the CLI has no argparse `choices`; names are resolved by `get_integrator`).

### Adding a new physics term (e.g. a PN correction)
- Keep the integrand construction explicit and readable — no hidden state.
- Give it an on/off flag in the relevant `model.py` function signature rather than making it unconditional, so existing callers and tests aren't silently changed (see how the O(c⁻⁴) term and Sun-J2 are wired).
- Add a validation note near the code: expected magnitude, and a sanity check for enabled vs disabled.

### Adding a new output column
- Add it to `OutputRequest` and `build_columns()` in `output.py`, gated behind its own flag — never unconditional.
- Wire the CLI flag in `cli.py` and document it in `examples/README.md`.

## Third-party dependencies

Versions below are what is installed in the project's `.venv` (Python 3.12.3; re-checked with `pip list` on 2026-10-15) — re-check before each formal deliverable snapshot. The `>=` minimums in `pyproject.toml`, and its `requires-python = ">=3.8"`, are floors that nothing in the code contradicts (every module defers its annotations with `from __future__ import annotations`), but only the versions below have been tested.

### Core (`pip install -e .`)

All four are imported by the library itself: numpy throughout; calcephpy for all ephemeris access (`ephem.py`); scipy for the sample-to-node interpolation in `chebycode/_stencil.py`; matplotlib for the plotting helpers in `chebycode/validation.py`, which load with `tempus.chebycode` — so the `chebyshev` subcommand needs it too.

| Package | Version | Download URL | License (SPDX) |
|---|---|---|---|
| numpy | 2.4.1 | https://pypi.org/project/numpy/2.4.1/ | BSD-3-Clause *(bundles 0BSD, MIT, Zlib, CC0-1.0 for vendored components)* |
| scipy | 1.17.0 | https://pypi.org/project/scipy/1.17.0/ | BSD-3-Clause |
| matplotlib | 3.10.8 | https://pypi.org/project/matplotlib/3.10.8/ | *no SPDX match — see fallback below* |
| calcephpy | 4.0.5 | https://pypi.org/project/calcephpy/4.0.5/ | CECILL-C |

**Matplotlib** ships a custom PSF-derived license text with no single SPDX identifier. Fallback reference:
- Name: Matplotlib License
- Version: as bundled with matplotlib 3.10.8 (`LICENSE` file in the distribution)
- URL: https://matplotlib.org

**calcephpy licensing** — the package's bundled `LICENSE` file states it is *"triple-licensed (CeCILL-C, CeCILL-B or CeCILL)"*: the licensee picks **one** to apply, they aren't stacked. TEMPUS uses calcephpy strictly as an unmodified runtime dependency (plain pip import, no forking/redistribution of its source), so **CeCILL-C** applies — the variant CALCEPH's own authors describe as *"close to the GNU LGPL"*, i.e. calling it as a library doesn't extend copyleft to the caller; only modifications to calcephpy itself would need to be shared. All three license texts still ship in the wheel (`COPYING_CECILL_C.LIB`, `COPYING_CECILL_B.LIB`, `COPYING_CECILL_V2.1.LIB`) if this choice needs revisiting.

### Notebooks (`pip install jupyter ipykernel astropy spiceypy`, as in the README)

The numbered notebooks otherwise use only the core packages. jupyter and ipykernel are also in the `dev` extra; astropy and spiceypy are in no extra.

| Package | Version | Download URL | License (SPDX) |
|---|---|---|---|
| jupyter | 1.1.1 | https://pypi.org/project/jupyter/1.1.1/ | BSD-3-Clause |
| ipykernel | 7.1.0 | https://pypi.org/project/ipykernel/7.1.0/ | BSD-3-Clause |
| astropy | 7.2.0 | https://pypi.org/project/astropy/7.2.0/ | BSD-3-Clause — notebook 00 only (epoch conversions) |
| spiceypy | 8.0.1 | https://pypi.org/project/spiceypy/8.0.1/ | MIT — notebook 00 only (leap-second / ET demo) |

### Tests (`dev` extra: `pip install -e ".[dev]"`)

| Package | Version | Download URL | License (SPDX) |
|---|---|---|---|
| pytest | 9.0.2 | https://pypi.org/project/pytest/9.0.2/ | MIT |

## Testing conventions

- Integrators and other pure-numeric code: unit tests against analytic functions, no external data required (see `tests/`).
- Anything requiring ephemeris/kernel files should skip gracefully when the data isn't present locally, not hard-fail (fetch data via `scripts/download_data.sh`).
- Sanity-check orders of magnitude when reviewing a new physics term (e.g. a v²-only drift lands at a very different scale than one including the potential term) rather than only checking exact values.
