# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempus.integrate import (
    cumulative_am10_uniform,
    cumulative_am4_uniform,
    cumulative_amk_uniform,
    get_integrator,
    PreparedGrid,
)


def test_am10_regression_sine_uniform():
    h = 0.5
    x = np.arange(0.0, 10.0 + h, h)
    rate = np.sin(x)
    out = cumulative_am10_uniform(rate, h)

    expected = np.array(
        [
            0.0000000000000000e00,
            1.1985638465105075e-01,
            4.5008051550407563e-01,
            9.0982200835706339e-01,
            1.3865201117144974e00,
            1.7634625044469070e00,
            1.9483605424878629e00,
            1.8959447375804248e00,
            1.6190483068310377e00,
            1.1854651535877814e00,
            6.9100655666257849e-01,
            2.6599984824749040e-01,
            1.4501493189678027e-02,
            -1.9129398809328985e-03,
            2.2077537472470993e-01,
            6.2804457106465916e-01,
            1.1201809458651084e00,
            1.5766923507187969e00,
            1.8858088723256938e00,
            1.9718480053808478e00,
            1.8137443693926887e00,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_am10_get_integrator_and_prepared_grid():
    h = 0.5
    x = np.arange(0.0, 10.0 + h, h)
    rate = np.sin(x)

    integ = get_integrator("am10")
    out_direct = cumulative_am10_uniform(rate, h)
    out_via_get = integ(rate, x)

    grid = PreparedGrid(x)
    out_via_grid = grid.integrate(rate, method="am10")

    np.testing.assert_allclose(out_via_get, out_direct, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(out_via_grid, out_direct, rtol=0.0, atol=1e-12)


def test_amk4_matches_am4_uniform():
    h = 0.5
    x = np.arange(0.0, 10.0 + h, h)
    rate = np.sin(x)

    out_am4 = cumulative_am4_uniform(rate, h)
    out_amk4 = cumulative_amk_uniform(rate, h, k=3)

    np.testing.assert_allclose(out_amk4, out_am4, rtol=0.0, atol=1e-12)
