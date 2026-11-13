# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempus.model import DAY_S, jd_grid, jd_parts_to_et
from tempus.output import build_columns


def test_jd_grid_split_and_uniform_et():
    jd_start = 2451545.0
    jd_step = 0.75
    jd_end = jd_start + 4.0 * jd_step

    grid = jd_grid(jd_start_int=2451545, jd_start_frac=np.float64(0.0),
                   jd_end_int=int(jd_end), jd_end_frac=np.float64(jd_end - int(jd_end)),
                   jd_step=jd_step)
    et = grid.et()

    jd = grid.jd_int.astype(np.float64) + grid.jd_frac
    expected_jd = jd_start + jd_step * np.arange(jd.size, dtype=np.float64)
    np.testing.assert_allclose(jd, expected_jd, rtol=0.0, atol=1e-12)

    expected_step_s = jd_step * DAY_S
    np.testing.assert_allclose(np.diff(et), expected_step_s, rtol=0.0, atol=1e-9)


def test_build_columns_uses_provided_jd_parts():
    grid = jd_grid(jd_start_int=2451545, jd_start_frac=np.float64(0.0),
                   jd_end_int=2451545,   jd_end_frac=np.float64(0.01),
                   jd_step=0.0025)
    et = grid.et()
    delta = np.linspace(0.0, 1.0, et.size, dtype=np.float64)

    cols = build_columns(
        jd_int=grid.jd_int,
        jd_frac=grid.jd_frac,
        tcb_minus_tcx=delta,
    )

    np.testing.assert_array_equal(cols["JD_int"], grid.jd_int)
    np.testing.assert_allclose(cols["JD_frac"], grid.jd_frac, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(cols["SecondsPastJ2000"], et, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(cols["TCB_minus_TCX"], delta, rtol=0.0, atol=0.0)

    # Split JD should be convertible back to ET without shape/type issues.
    et_back = jd_parts_to_et(cols["JD_int"], cols["JD_frac"])
    np.testing.assert_allclose(et_back, cols["SecondsPastJ2000"], rtol=0.0, atol=1e-6)
