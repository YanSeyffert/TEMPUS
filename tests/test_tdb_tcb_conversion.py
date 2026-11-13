# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_tdb_tcb_conversion.py
# IAU 2006 Res. B3 TDB<->TCB split-JD grid conversion.

import numpy as np
from numpy.testing import assert_allclose

from tempus.model import (
    DAY_S,
    L_B,
    TDB0_S,
    TimeGrid,
    jd_grid,
    jd_parts_to_et,
    jd_tcb_to_tdb,
    jd_tdb_to_tcb,
)
from tempus.integrate import PreparedGrid

T0_INT, T0_FRAC = 2443144, 0.5003725


def _grid_of(int_part: int, frac_part: float) -> TimeGrid:
    return TimeGrid(np.array([int_part], dtype=np.int64), np.array([frac_part], dtype=np.float64))


def test_anchor_at_t0():
    # At JD_TCB = T0 the linear term vanishes: TDB = TCB + TDB0.
    tdb = jd_tcb_to_tdb(_grid_of(T0_INT, T0_FRAC))
    assert tdb.jd_int[0] == T0_INT
    assert_allclose(tdb.jd_frac[0], T0_FRAC + TDB0_S / DAY_S, rtol=0.0, atol=1e-16)


def test_round_trip_over_millennium():
    # +-600 years around J2000; round trip must return the input to ~1 ULP of frac.
    grid = jd_grid(2231545, 0.25, 2671545, 0.25, 4999.0)
    back = jd_tcb_to_tdb(jd_tdb_to_tcb(grid))
    np.testing.assert_array_equal(back.jd_int, grid.jd_int)
    assert_allclose(back.jd_frac, grid.jd_frac, rtol=0.0, atol=3e-16)


def test_offset_magnitude_at_j2000():
    # Literature value: TCB - TDB ~ +11.25 s at J2000.
    tcb = jd_tdb_to_tcb(_grid_of(2451545, 0.0))
    diff_s = (float(tcb.jd_int[0] - 2451545) + tcb.jd_frac[0]) * float(DAY_S)
    assert 11.2 < diff_s < 11.3


def test_uniform_tdb_grid_maps_to_uniform_tcb_grid():
    h = 0.5
    grid = jd_grid(2451545, 0.0, 2451945, 0.0, h)
    tcb = jd_tdb_to_tcb(grid)
    # Spacing measured in split arithmetic (int diff exact, frac diff ~1 ULP);
    # recombining full JDs first would drown the signal in ulp(2.45e6) noise.
    step = np.diff(tcb.jd_int).astype(np.float64) + np.diff(tcb.jd_frac)
    assert_allclose(step, h / (1.0 - float(L_B)), rtol=0.0, atol=1e-15)


def test_converted_grid_passes_integrator_uniformity_check():
    # Per-point conversion perturbs the step only at the ULP level; the
    # fixed-step integrators' uniformity check must still accept the grid.
    grid = jd_grid(2451545, 0.0, 2455545, 0.0, 0.1)
    tcb = jd_tdb_to_tcb(grid)
    PreparedGrid(jd_parts_to_et(tcb.jd_int, tcb.jd_frac))  # must not raise
