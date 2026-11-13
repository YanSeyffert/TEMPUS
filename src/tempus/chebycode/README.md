# chebycode

Piecewise Chebyshev polynomial fitting for high-precision ephemeris and time-series data.

The time axis is Julian Date (JD), supplied as a two-part split `(jd_int, jd_frac)`. The two parts are never merged inside the library — only file I/O does so — preserving full float64 precision across the ~2.4 × 10⁶-day JD range.

## Quick start

```python
import numpy as np
from chebycode import ChebyshevSeries

# jd_int, jd_frac, y, dy come from your data source (e.g. CALCEPH, SPICE, …)
# jd_int : (N,) int64   — integer day numbers
# jd_frac : (N,) float64 — fractional days  (jd_int + jd_frac = full JD)
# y       : (N,) or (N, D) float64 — sampled values
# dy      : (N,) or (N, D) float64 — rates in units/day  (optional)

cs = ChebyshevSeries.from_samples(
    jd_int, jd_frac, y, dy=dy,
    granule_len=4, order=12,
)

# Evaluate at query JDs
vals = cs.evaluate(jd_int_q, jd_frac_q)        # (M,) or (M, D)
dots = cs.evaluate_deriv(jd_int_q, jd_frac_q)  # (M,) or (M, D)  [units per day]

# Save and reload
cs.to_asc("fit.asc")
cs2 = ChebyshevSeries.from_asc("fit.asc")
```

`jd_int` and `jd_frac` can be any consistent split as long as `jd_int + jd_frac` is strictly ascending. When only a combined JD is available, pass it as `jd_int` and omit `jd_frac` (or pass `jd_frac=0`).

If `dy` is supplied it must be in units per JD day. Convert rates from other time bases before fitting:

```python
dy_day = dy_per_second * 86400.0
```

### Recommended settings

| Use case             | `granule_len` | `order` | `dist` |
|----------------------|:---:|:---:|:---:|
| Moon position        | 4   | 12  | 1   |
| Orientation angles   | 8   | 11  | 2   |
| Scalar time series   | 4   | 11  | 3   |

### Multi-dimensional data (e.g. 3D position)

Stack components into a `(N, D)` array — all dimensions share one `ChebyshevSeries` object:

```python
pos = np.column_stack([x, y, z])    # (N, 3)
vel = np.column_stack([vx, vy, vz]) # (N, 3)

cs = ChebyshevSeries.from_samples(
    jd_int, jd_frac, pos, dy=vel,
    granule_len=4, order=12, dist=1,
)
cs.evaluate(jd_int_q, jd_frac_q)        # (M, 3)
cs.evaluate_deriv(jd_int_q, jd_frac_q)  # (M, 3)
```

### Parallel fitting

For very long time spans, granules can be fitted concurrently. NumPy/SciPy release the GIL during heavy array work, so threads genuinely run in parallel:

```python
cs = ChebyshevSeries.from_samples(..., n_workers=8)
```

In practice, NumPy's internal BLAS threads already saturate the CPU for typical `order=12` granules, so gains from `n_workers > 1` are dataset-dependent.

## File I/O

The native format is a human-readable ASCII file with `.17g`-precision coefficients:

```python
cs.to_asc("fit.asc", title="Moon geocentric position [AU]")
cs2 = ChebyshevSeries.from_asc("fit.asc")
```

`from_asc` also reads INPOP native `.asc` files and makes a best-effort attempt at other ASCII tabular formats (1-D only).

**Format note:** the first line of every file written by `to_asc` starts with `ChebyshevSeries`. If you supply a custom `title` that does not already start with it, the prefix is prepended automatically so the loader can identify the file.

## Diagnostics

```python
# Print a formatted continuity + truncation report to stdout
cs.report()

# Compare against reference state vectors and show residual plots
# pv_ref shape: (N, 6) — values in cols 0:3, derivatives in cols 3:6
cs.validate(jd_int, jd_frac, pv_ref, kind="position")    # or kind="orientation"
# -> dict with max_abs_error, rms_error, mean_error
```

## How it works

### Granules

Fitting one polynomial over a span of decades would require an impractically high degree. Instead, the time span is cut into short fixed-length windows called **granules**, each fitted independently. Only the coefficients `c[0]…c[order]` are stored per granule — for 3D data (X, Y, Z) that is 13 × 3 = 39 floats at the recommended settings.

### Polynomial basis

Each granule's polynomial is a weighted sum of Chebyshev basis functions T_j on the normalised coordinate τ ∈ [−1, +1]:

```
f(τ) = c[0]·T_0(τ)  +  c[1]·T_1(τ)  +  …  +  c[n]·T_n(τ)
```

The basis functions follow a simple three-term recurrence: T_0 = 1, T_1 = τ, T_{j+1} = 2τ·T_j − T_{j-1}. For smooth data the coefficients shrink geometrically with degree — the last few coefficients serve as a built-in convergence indicator, and evaluation is fast and numerically stable.

### Getting the coefficients

**Step 1 — pick `ncoeff` nodes inside the granule.**
`ncoeff = order + 1` collocation points (nodes) are placed at Chebyshev-optimal positions clustered toward the granule edges. This keeps the linear system well-conditioned and avoids Runge-type oscillations.

**Step 2 — estimate the function value at each node (stencil).**
Raw input samples rarely fall on the node positions. For each node the `npoints` nearest raw samples form a local sliding window and a small polynomial is fitted through them to get an estimate there.

If derivatives `dy` are also supplied, each sample contributes two constraints instead of one — doubling the effective polynomial degree of the local fit:

- **Hermite** (`dy` given): `npoints` samples → degree `2·npoints − 1` local polynomial
- **Lagrange** (no `dy`): `npoints` samples → degree `npoints − 1` local polynomial

With `npoints=6` and derivatives, 12 constraints yield a degree-11 local polynomial — enough to estimate the node value accurately even from sparse sampling.

**Step 3 — solve for the coefficients.**
Evaluating each basis polynomial at each node position gives the Chebyshev Vandermonde matrix **M**. The coefficients follow from:

```
c = M⁻¹ · f
```

**M** depends only on node positions and order, not on the data. So **M⁻¹** is computed once via SVD (built in `longdouble` for better conditioning), cached per `(ncoeff, dist)` pair, and reused for every granule. The per-granule cost is one matrix–vector multiply.

**Convergence.** Because of how Chebyshev polynomials work, the coefficients of a smooth function decay geometrically with degree. Plotting |c[j]| vs j on a log scale should reach machine precision before j = order. If they don't decay, the order is too low or the granule is too long. `cs.report()` shows this check.

### Node placement (`dist`)

Where exactly to place the nodes affects conditioning. Edge-clustered nodes outperform evenly-spaced ones at high degree.

| `dist` | distribution | behaviour |
|:---:|---|---|
| 1 | C¹ Gauss-Lobatto | adjacent granules share boundary nodes and match both value and slope there; requires `dy` |
| 2 | Gauss-Lobatto | endpoints ±1 included as nodes; no derivative constraint |
| 3 | Chebyshev-Gauss | interior roots of T_ncoeff; endpoints not included |

For `dist=1` rows 1 and `ncoeff−1` of **M** are replaced by derivative rows before inversion, encoding the C¹ boundary conditions directly into the coefficient solve.

### Evaluation

Given a query JD:

**Step 1 — find the granule.** Binary search on stored granule edges → locate gs ≤ JD < ge.

**Step 2 — rescale to τ ∈ [−1, +1].**

```
τ = ( 2·JD − gs − ge ) / ( ge − gs )
```

JD values are large (~2.4 × 10⁶), so the numerator suffers catastrophic cancellation if computed naively. Supplying JD as integer + fractional parts lets the large integer portions cancel exactly in integer arithmetic, with only the small remainders contributing float rounding.

**Step 3 — evaluate via Clenshaw recurrence.** Computing each T_j(τ) explicitly and summing would accumulate rounding errors at high degree. Clenshaw's algorithm runs a backward recurrence that evaluates the full sum in O(n) steps with no precision loss:

```
b_j = 2τ · b_{j+1} − b_{j+2} + c[j]    (b_{n+1} = b_{n+2} = 0)
result = τ · b_1 − b_2 + c[0]
```

**Derivative** d/dJD is computed analytically by a companion forward pass propagating T_j(τ) and T'_j(τ) together, then scaling back: `d/dJD = (d/dτ) · 2 / granule_len`. No finite differences.

All query points hitting the same granule are evaluated simultaneously via NumPy broadcasting.

## Precision

The Vandermonde matrix and derivative rows are built in `np.longdouble` (80-bit extended on x86-64 Linux/macOS; equivalent to float64 on Windows) before SVD inversion. This keeps M⁻¹ accurate for orders up to ~40 even in float64 storage. The split-JD argument reduction keeps τ accurate regardless of longdouble availability.

## Module structure

| module | role |
|---|---|
| `series.py` | `ChebyshevSeries` — the top-level user API |
| `io.py` | `save_ascii` / `load_ascii` and format dispatchers |
| `validation.py` | continuity/truncation diagnostics, residual plots |
| `_granule.py` | `Granule` dataclass; `fit_granule`, `evaluate_batch` |
| `_basis.py` | M⁻¹ cache, `tau_from_jd`, `clenshaw_batch` |
| `_nodes.py` | collocation node placement |
| `_stencil.py` | local Hermite/Lagrange interpolation to nodes |

## API reference

| API | description |
|---|---|
| `ChebyshevSeries.from_samples(jd_int, jd_frac, y, dy, granule_len, order, dist, npoints, n_workers)` | fit from samples |
| `ChebyshevSeries.from_asc(path)` | load from ASCII file |
| `.evaluate(jd_int, jd_frac)` | evaluate values — returns `(M,)` or `(M, D)` |
| `.evaluate_deriv(jd_int, jd_frac)` | evaluate d/dJD — same shape |
| `.to_asc(path, title)` | save to ASCII |
| `.report()` | print continuity and truncation summary |
| `.validate(jd_int, jd_frac, pv_ref, kind, label)` | residual stats and plots |
| `.granule_edges` | `(G+1,)` float64 boundary JDs |
| `.coefficients` | list of G arrays, each `(order+1,)` or `(order+1, D)` |
| `.order`, `.ndim`, `.n_granules` | scalar properties |
| `.jd_start`, `.jd_end` | time range as combined float64 |
