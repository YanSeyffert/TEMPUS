# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# tests/test_model_tcx_tcy.py
# Delta_TCX_TCY BCRS subtraction: sign conventions, per-leg ext lists, boost term.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tempus import model

MOON, EARTH, SUN, JUP = 301, 399, 10, 5


def _fake_delta_tcx(calls):
    def fake(eph, *, jd_int, jd_frac, target, center, ext_bodies, gm_provider, **kw):
        calls.append({"target": target, "center": center, "ext": list(ext_bodies)})
        if target == MOON:
            return np.array([0.0, 4.0]), np.array([1.0, 1.5])
        return np.array([0.0, 1.0]), np.array([1.0, 1.2])
    return fake


def test_signs_match_notes_convention(monkeypatch):
    # delta_X = TCB - TCX  =>  TC_center - TC_target = d_target - d_center,
    # rate = r_center - r_target (r = dTCX/dTCB).
    calls = []
    monkeypatch.setattr(model, "Delta_TCX", _fake_delta_tcx(calls))
    delta, rate = model.Delta_TCX_TCY(
        None, jd_int=np.array([0, 1]), jd_frac=np.zeros(2),
        target=MOON, center=EARTH, ext_bodies=[SUN, JUP, MOON, EARTH], gm_provider=None,
    )
    assert_allclose(delta, [0.0, 3.0])
    assert_allclose(rate, [0.0, -0.3])


def test_per_leg_ext_lists_exclude_own_body_only(monkeypatch):
    calls = []
    monkeypatch.setattr(model, "Delta_TCX", _fake_delta_tcx(calls))
    model.Delta_TCX_TCY(
        None, jd_int=np.array([0, 1]), jd_frac=np.zeros(2),
        target=MOON, center=EARTH, ext_bodies=[SUN, JUP, MOON, EARTH], gm_provider=None,
    )
    by_target = {c["target"]: c for c in calls}
    assert by_target[MOON]["ext"] == [SUN, JUP, EARTH]   # Earth stays external for the Moon leg
    assert by_target[EARTH]["ext"] == [SUN, JUP, MOON]   # Moon stays external for the Earth leg
    assert all(c["center"] == 0 for c in calls)          # both legs are SSB-centered


def test_missing_other_body_is_added(monkeypatch):
    calls = []
    monkeypatch.setattr(model, "Delta_TCX", _fake_delta_tcx(calls))
    model.Delta_TCX_TCY(
        None, jd_int=np.array([0, 1]), jd_frac=np.zeros(2),
        target=MOON, center=EARTH, ext_bodies=[SUN], gm_provider=None,
    )
    by_target = {c["target"]: c for c in calls}
    assert by_target[MOON]["ext"] == [SUN, EARTH]
    assert by_target[EARTH]["ext"] == [SUN, MOON]


def test_same_body_raises():
    with pytest.raises(ValueError):
        model.Delta_TCX_TCY(
            None, jd_int=np.array([0]), jd_frac=np.zeros(1),
            target=MOON, center=MOON, ext_bodies=[SUN], gm_provider=None,
        )


class _FakeEph:
    """state_jd stub: center-relative target position and center BCRS velocity."""

    def state_jd(self, jd_int, jd_frac, target, center=0):
        n = np.asarray(jd_int).size
        c2 = model.C ** 2
        if center != 0:  # target relative to center: position (c^2, 2*c^2, 0) m
            return np.tile([c2, 2.0 * c2, 0.0, 0.0, 0.0, 0.0], (n, 1))
        # center relative to SSB: velocity (3, 4, 5) m/s
        return np.tile([0.0, 0.0, 0.0, 3.0, 4.0, 5.0], (n, 1))


def test_boost_term_dot_product():
    # v_center . r_tc / c^2 = (3*c^2 + 4*2c^2) / c^2 = 11 s
    out = model.boost_term(
        _FakeEph(), jd_int=np.array([0, 1, 2]), jd_frac=np.zeros(3),
        target=MOON, center=EARTH,
    )
    assert_allclose(out, 11.0)
