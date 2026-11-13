# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempus.integrate import get_integrator
from tempus.model import DAY_S, JD_J2000, Delta_TCX, _build_context, _extend_time_for_am_startup, _rate_o2, jd_parts_to_et


class _FakeEph:
    def __init__(self, w: float):
        self.w = w

    def timescale(self):
        return "TCB"

    def state_jd(self, jd_int, jd_frac, target, center=0):  # noqa: ARG002
        et = jd_parts_to_et(jd_int, jd_frac)
        if int(target) == 399:
            x = np.zeros((et.size, 3), dtype=np.float64)
            v = np.zeros((et.size, 3), dtype=np.float64)
            return np.hstack((x, v))

        a = 3.0e10
        wt = self.w * et
        x = np.column_stack(
            (
                a * (1.1 + 0.7 * np.cos(wt)),
                a * 0.7 * np.sin(wt),
                np.zeros(et.size, dtype=np.float64),
            )
        )
        v = np.column_stack(
            (
                -a * 0.7 * self.w * np.sin(wt),
                a * 0.7 * self.w * np.cos(wt),
                np.zeros(et.size, dtype=np.float64),
            )
        )
        return np.hstack((x, v))


def test_delta_tcx_am10_uses_preroll_startup():
    step_s = 0.75 * 86400.0
    et = np.arange(0.0, 300.0 * step_s, step_s)
    jd = np.longdouble(JD_J2000) + et.astype(np.longdouble) / np.longdouble(DAY_S)
    jd_int = np.floor(jd).astype(np.int64)
    jd_frac = (jd - jd_int.astype(np.longdouble)).astype(np.float64)
    w = 2.0 * np.pi / (20.0 * step_s)
    eph = _FakeEph(w=w)
    mu_sun = 1.32712440018e20
    gm_provider = lambda _: mu_sun

    out, _ = Delta_TCX(
        eph=eph,
        jd_int=jd_int,
        jd_frac=jd_frac,
        target=399,
        center=0,
        ext_bodies=[10],
        gm_provider=gm_provider,
        integrator="am10",
    )

    jd_int_ext, jd_frac_ext, start_idx = _extend_time_for_am_startup(jd_int, jd_frac, "am10", None)
    et_ext = jd_parts_to_et(jd_int_ext, jd_frac_ext)
    ctx_ext = _build_context(
        eph,
        jd_int=jd_int_ext,
        jd_frac=jd_frac_ext,
        target=399,
        center=0,
        ext_bodies=[10],
        gm_provider=gm_provider,
    )
    rate_ext = -_rate_o2(ctx_ext)
    ref_ext = get_integrator("am10")(rate_ext, et_ext)
    ref = ref_ext[start_idx:] - ref_ext[start_idx]
    np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-12)

    # Old behavior (no pre-roll) has a visible startup offset for this setup.
    ctx_short = _build_context(
        eph,
        jd_int=jd_int,
        jd_frac=jd_frac,
        target=399,
        center=0,
        ext_bodies=[10],
        gm_provider=gm_provider,
    )
    rate_short = -_rate_o2(ctx_short)
    old_like = get_integrator("am10")(rate_short, et)
    assert float(np.max(np.abs(out - old_like))) > 1e-7
