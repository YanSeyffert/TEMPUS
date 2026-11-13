# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from tempus.chebycode import ChebyshevSeries


def test_orientation_angles_with_large_offset_preserve_trend_when_dy_is_per_jd():
    jd = np.linspace(2460000.5, 2460050.5, 501, dtype=np.float64)
    jd1 = np.floor(jd).astype(np.int64)
    jd2 = jd - jd1.astype(np.float64)
    x = jd - jd[0]

    angles = np.column_stack([
        0.08 + 0.01 * x + 0.001 * np.sin(0.1 * x),
        0.5 + 0.002 * np.cos(0.05 * x),
        3.0 + 0.05 * np.sin(0.08 * x),
    ])
    dangles_day = np.column_stack([
        0.01 + 0.0001 * np.cos(0.1 * x),
        -0.002 * 0.05 * np.sin(0.05 * x),
        0.05 * 0.08 * np.cos(0.08 * x),
    ])

    series = ChebyshevSeries.from_samples(
        jd1,
        jd2,
        angles,
        dangles_day,
        granule_len=8,
        order=10,
        orientation=True,
    )
    fit = series.evaluate(jd1, jd2)

    assert_allclose(fit, angles, atol=1e-12)
    assert_allclose(np.mean(fit, axis=0), np.mean(angles, axis=0), atol=1e-12)
    assert_allclose(np.polyfit(x, fit[:, 0], 1), np.polyfit(x, angles[:, 0], 1), atol=1e-12)
