# Changelog

Notable changes to TEMPUS. Any change to numerical results is listed, so you know when products need to be regenerated.

## [1.1.0] - 2026-11-16

### Added
- `tempus integrate --zero` writes the integrated offset into the `.csv` already zeroed at the IAU 1977 origin (1977-01-01 00:00:00 TAI) on the grid's own axis, or at another split JD with `--zero INT FRAC`. The value at that epoch comes from a local polynomial through the 12 nearest samples. determining the needed shift. Without `--zero` the output is unchanged: the offset is 0 at the first sample.
- The `.run.json` always records where the integrated offset is zero (epoch, timescale, origin, method, used shift): the `--zero` epoch, or otherwise the first sample. `tempus chebyshev` carries that zero into the `.asc` title with its axis, marked as the IAU 1977 origin or as custom, e.g. `zero=TDB:JD2443144+0.5003724992418982(IAU1977)`.
- `tempus.model.jd_origin_1977()` and `tempus.model.value_at_epoch()`, used by both `integrate --zero` and `chebyshev --anchor`.

### Fixed
- `tempus chebyshev --anchor` without values put the 1977 origin at JD 2443144.5003725 on every axis. On a TDB axis the origin is at T0 + TDB0/86400, 65.5 µs earlier, and the anchor now goes there; the axis is read from `in_timescale` in the `.run.json`.
  **Numerical change:** series anchored this way on TDB runs move by about 1 ps. TCB runs are unchanged. Without `in_timescale`, bare `--anchor` now stops with an error.
- The `.asc` title records the run's actual argument timescale (`TDB` or `TCB`; 1.0.0 always wrote `native`) and the anchor epoch at full precision. The printed shift has 17 significant digits instead of 9 decimals.

## [1.0.0] - 2026-11-13

First public release (Zenodo DOI 10.5281/zenodo.24199183; IDDN.FR.001.460016.000.S.P.2026.000.31235). Model and validation: [arXiv:2609.08979](https://arxiv.org/abs/2609.08979).
