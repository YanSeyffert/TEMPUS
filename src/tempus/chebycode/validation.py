# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

"""
validation.py — Quality assurance and visualisation for ChebyshevSeries2.

Two groups of functions:

  Numerical diagnostics
  ---------------------
  continuity_report        — C⁰ and C¹ jump at each granule boundary
  truncation_error_estimate — geometric decay of trailing Chebyshev coefficients
  compare_against_samples  — max / RMS error vs a reference dataset

  Validation plots
  ----------------
  plot_compare_p        — position residuals X, Y, Z  [AU]
  plot_compare_v        — velocity residuals Vx, Vy, Vz  [AU/day]
  plot_compare_angle    — Euler angle residuals φ, θ, ψ  [rad]
  plot_compare_angledot — Euler angle-rate residuals φ̇, θ̇, ψ̇  [rad/day]

  All plot functions expect arrays of shape (N, 6): the first three columns
  are values, the next three are their time derivatives — the format returned
  by ILuRFState.state() or by stacking evaluate() + evaluate_deriv().
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._basis import clenshaw_batch, clenshaw_deriv_batch

_LOG_FLOOR = np.float64(1e-300)   # guard against log(0)
JD_J2000   = 2451545.0


# ===========================================================================
# Numerical diagnostics
# ===========================================================================

def continuity_report(series) -> dict[str, float]:
    """
    Check C⁰ and C¹ continuity at each internal granule boundary.

    Evaluates value and derivative from the left (τ = +1 of granule g) and
    from the right (τ = -1 of granule g+1) at every junction.  Derivatives
    are scaled to physical units (per JD) via dτ/dJD = 2 / granule_len.

    Returns
    -------
    dict with keys max_abs_jump, rms_jump, max_abs_deriv_jump, rms_deriv_jump.
    """
    if series.n_granules < 2:
        return {"max_abs_jump": 0.0, "rms_jump": 0.0,
                "max_abs_deriv_jump": 0.0, "rms_deriv_jump": 0.0}

    tau_right = np.array([1.0])
    tau_left  = np.array([-1.0])
    jumps, deriv_jumps = [], []

    for i in range(series.n_granules - 1):
        g_r, g_l = series.granules[i], series.granules[i + 1]

        jumps.append(
            clenshaw_batch(g_l.coeffs, tau_left ).ravel() -
            clenshaw_batch(g_r.coeffs, tau_right).ravel()
        )
        # Scale dS/dτ → dS/dJD:  multiply by dτ/dJD = 2 / granule_len
        deriv_jumps.append(
            clenshaw_deriv_batch(g_l.coeffs, tau_left ).ravel() * (2.0 / (g_l.t_end - g_l.t_start)) -
            clenshaw_deriv_batch(g_r.coeffs, tau_right).ravel() * (2.0 / (g_r.t_end - g_r.t_start))
        )

    def _stats(rows):
        a = np.abs(np.asarray(rows, dtype=np.float64))
        return float(np.max(a)), float(np.sqrt(np.mean(a**2)))

    max_j,  rms_j  = _stats(jumps)
    max_dj, rms_dj = _stats(deriv_jumps)
    return {
        "max_abs_jump":       max_j,
        "rms_jump":           rms_j,
        "max_abs_deriv_jump": max_dj,
        "rms_deriv_jump":     rms_dj,
    }


def truncation_error_estimate(series, n_tail: int = 4) -> dict:
    """
    Estimate the Chebyshev truncation error from trailing coefficient decay.

    For a well-resolved series the coefficients decay geometrically:
        |c_j| ≈ A · r^j,   r < 1

    The truncation error is bounded by the geometric tail sum:
        ε ≤ |c_n| · r / (1 - r)   (when r < 1)

    The decay rate r is estimated by fitting a line to log|c_j| over the last
    n_tail coefficients.

    Returns
    -------
    dict with keys last_coeff, decay_rate, tail_estimate, converged.
    """
    if n_tail < 2:
        raise ValueError("n_tail must be >= 2")

    n_granules   = series.n_granules
    last_coeff_g = np.empty(n_granules)
    decay_rate_g = np.empty(n_granules)
    tail_est_g   = np.empty(n_granules)

    for g, c in enumerate(series.coefficients):
        c   = np.asarray(c, dtype=np.float64)
        amp = np.abs(c).max(axis=1) if c.ndim == 2 else np.abs(c)

        n          = len(amp) - 1
        tail_slice = amp[max(0, n - n_tail + 1):]
        js         = np.arange(len(tail_slice), dtype=np.float64)
        log_amp    = np.log(np.maximum(tail_slice, _LOG_FLOOR))
        r          = float(np.exp(np.polyfit(js, log_amp, 1)[0]))

        last = float(amp[n])
        last_coeff_g[g] = last
        decay_rate_g[g] = r
        tail_est_g[g]   = last * r / (1.0 - r) if r < 1.0 else last

    return {
        "last_coeff":    float(np.max(last_coeff_g)),
        "decay_rate":    float(np.median(decay_rate_g)),
        "tail_estimate": float(np.max(tail_est_g)),
        "converged":     bool(np.all(decay_rate_g < 1.0)),
    }


def compare_against_samples(
    series,
    jd_int:  np.ndarray,
    jd_frac: np.ndarray,
    y:       np.ndarray,
) -> dict[str, float]:
    """
    Compare the fitted series against reference samples at given JD values.

    Parameters
    ----------
    series  : ChebyshevSeries2
    jd_int  : (N,) int64 integer parts of Julian dates
    jd_frac : (N,) float64 fractional parts of Julian dates
    y       : (N,) or (N, ndim) reference values

    Returns
    -------
    dict with keys max_abs_error, rms_error, mean_error.
    """
    pred = series.evaluate(jd_int, jd_frac)
    ref  = np.asarray(y, dtype=np.float64)

    if ref.ndim  == 1: ref  = ref[:, None]
    if pred.ndim == 1: pred = pred[:, None]

    diff     = pred - ref
    abs_diff = np.abs(diff)
    return {
        "max_abs_error": float(np.max(abs_diff)),
        "rms_error":     float(np.sqrt(np.mean(abs_diff**2))),
        "mean_error":    float(np.mean(diff)),
    }


# ===========================================================================
# Formatted report helpers
# ===========================================================================

def print_report(series) -> None:
    """
    Print a formatted summary of continuity and truncation diagnostics.

    No arguments needed — operates entirely on the fitted series.
    """
    cr = continuity_report(series)
    tr = truncation_error_estimate(series)

    w = 50
    print(repr(series))
    print("─" * w)
    print(f"  Continuity   C⁰ max jump  : {cr['max_abs_jump']:.2e}")
    print(f"               C¹ max jump  : {cr['max_abs_deriv_jump']:.2e}")
    print(f"  Truncation   last coeff   : {tr['last_coeff']:.2e}")
    print(f"               decay-rate   : {tr['decay_rate']:.2f}")
    print(f"               converged    : {tr['converged']}")
    print("─" * w)


def validate_series(
    series,
    jd_int:        np.ndarray,
    jd_frac:       np.ndarray,
    pv_ref:        np.ndarray,
    *,
    kind:          str   = "position",
    label:         str   = "",
    x_tick:        float = 2,
    show_overview: bool  = False,
    unwrap:        bool  = False,
) -> dict[str, float]:
    """
    Compare the fitted series against reference data and show residual plots.

    Evaluates the series at the given JD values, computes errors against
    pv_ref[:, :ndim], and shows residual plots for both values and derivatives.

    Parameters
    ----------
    jd_int, jd_frac : query times in split form
    pv_ref          : (N, 6) reference array — values in cols 0:3, derivatives
                      in cols 3:6 (same layout as evaluate + evaluate_deriv stacked)
    kind            : 'position'    → plot_compare_p + plot_compare_v
                      'orientation' → plot_compare_angle + plot_compare_angledot
    label           : plot title prefix
    x_tick          : x-axis tick spacing in years
    show_overview   : if True, also plot raw values alongside residuals
    unwrap          : unwrap angles before differencing (orientation only)

    Returns
    -------
    dict with keys max_abs_error, rms_error, mean_error (values only, not derivatives)
    """
    pv_ref = np.asarray(pv_ref, dtype=np.float64)
    ndim   = series.ndim

    pos = series.evaluate(jd_int, jd_frac)
    vel = series.evaluate_deriv(jd_int, jd_frac)
    if pos.ndim == 1:
        pos = pos[:, None]
        vel = vel[:, None]
    pv_fit = np.column_stack([pos, vel])

    # Error stats on values only
    stats = compare_against_samples(series, jd_int, jd_frac, pv_ref[:, :ndim])
    print(f"Samples   max  error : {stats['max_abs_error']:.5e}")
    print(f"          RMS  error : {stats['rms_error']:.5e}")
    print(f"          mean error : {stats['mean_error']:.5e}")

    # Combined JD for plot x-axis (precision is sufficient for display)
    jd_combined = jd_int.astype(np.float64) + jd_frac
    lbl = label or {"position": "Position", "orientation": "Orientation", "time": "Time"}.get(kind, kind)

    if kind == "position":
        plot_compare_p(jd_combined, pv_fit, pv_ref,
                       label=lbl, x_tick=x_tick, show_overview=show_overview)
        plot_compare_v(jd_combined, pv_fit, pv_ref,
                       label=lbl, x_tick=x_tick, show_overview=show_overview)
    elif kind == "orientation":
        plot_compare_angle(jd_combined, pv_fit, pv_ref,
                           label=lbl, x_tick=x_tick,
                           show_overview=show_overview, unwrap=unwrap)
        plot_compare_angledot(jd_combined, pv_fit, pv_ref,
                              label=lbl, x_tick=x_tick, show_overview=show_overview)
    elif kind == "time":
        plot_compare_time(jd_combined, pv_fit, pv_ref,
                          label=lbl, x_tick=x_tick, show_overview=show_overview)
        if pv_ref.shape[1] > 1:
            plot_compare_timedot(jd_combined, pv_fit, pv_ref,
                                 label=lbl, x_tick=x_tick, show_overview=show_overview)
    else:
        raise ValueError(f"kind must be 'position', 'orientation', or 'time', got {kind!r}")


# ===========================================================================
# Validation plots
# ===========================================================================

def jd_to_yr(jd: np.ndarray) -> np.ndarray:
    """Convert Julian Date to civil year (J2000 reference)."""
    return (np.asarray(jd) - JD_J2000) / 365.25 + 2000.0


def plot_compare_p(
    jds,
    pv_queried,
    pv_reference,
    label:         str  = "Body",
    x_tick:        float = 2,
    show_overview: bool  = False,
) -> None:
    """
    Plot position residuals (X, Y, Z) between two (N, 6) arrays.

    Parameters
    ----------
    jds          : (N,) Julian dates
    pv_queried   : (N, 6) — queried values (e.g. from ChebyshevSeries2.evaluate)
    pv_reference : (N, 6) — reference values (e.g. raw integration output)
    label        : title prefix
    x_tick       : major tick spacing in years
    show_overview: if True, also plot raw values before the residual figure
    """
    _plot_compare_pv(
        jds, pv_queried, pv_reference,
        label=label, x_tick=x_tick,
        cols=[0, 1, 2], comp_labels=["X", "Y", "Z"], units="AU",
        show_overview=show_overview,
    )


def plot_compare_v(
    jds,
    pv_queried,
    pv_reference,
    label:         str   = "Body",
    x_tick:        float = 2,
    show_overview: bool  = False,
) -> None:
    """
    Plot velocity residuals (Vx, Vy, Vz) between two (N, 6) arrays.
    """
    _plot_compare_pv(
        jds, pv_queried, pv_reference,
        label=label, x_tick=x_tick,
        cols=[3, 4, 5], comp_labels=["Vx", "Vy", "Vz"], units="AU/day",
        show_overview=show_overview,
    )


def plot_compare_angle(
    jds,
    orient_queried,
    orient_reference,
    label:         str   = "Euler Body",
    x_tick:        float = 2,
    show_overview: bool  = False,
    unwrap:        bool  = False,
) -> None:
    """
    Plot Euler angle residuals (φ, θ, ψ) between two (N, 6) arrays.

    Parameters
    ----------
    unwrap : if True, unwrap angles before differencing to remove 2π jumps
    """
    a = np.array(orient_queried,   copy=True)
    b = np.array(orient_reference, copy=True)
    if unwrap:
        a[:, :3] = np.unwrap(a[:, :3], axis=0)
        b[:, :3] = np.unwrap(b[:, :3], axis=0)
    _plot_compare_pv(
        jds, a, b,
        label=label, x_tick=x_tick,
        cols=[0, 1, 2], comp_labels=["Phi", "Theta", "Psi"], units="Radians",
        show_overview=show_overview,
    )


def plot_compare_angledot(
    jds,
    orient_queried,
    orient_reference,
    label:         str   = "Euler Body",
    x_tick:        float = 2,
    show_overview: bool  = False,
    units:         str   = "Radians/day",
) -> None:
    """
    Plot Euler angle-rate residuals (φ̇, θ̇, ψ̇) between two (N, 6) arrays.
    """
    _plot_compare_pv(
        jds, orient_queried, orient_reference,
        label=label, x_tick=x_tick,
        cols=[3, 4, 5], comp_labels=["PhiDot", "ThetaDot", "PsiDot"], units=units,
        show_overview=show_overview,
    )


def plot_compare_time(
    jds,
    pv_queried,
    pv_reference,
    label:         str   = "Time",
    x_tick:        float = 2,
    show_overview: bool  = False,
    units:         str   = "s",
) -> None:
    """Plot scalar time-series residuals (e.g. TCL offset)."""
    _plot_compare_pv(
        jds, pv_queried, pv_reference,
        label=label, x_tick=x_tick,
        cols=[0], comp_labels=["TCL"], units=units,
        show_overview=show_overview,
    )


def plot_compare_timedot(
    jds,
    pv_queried,
    pv_reference,
    label:         str   = "Time",
    x_tick:        float = 2,
    show_overview: bool  = False,
    units:         str   = "s/day",
) -> None:
    """Plot scalar time-series derivative residuals (e.g. dTCL/dt)."""
    _plot_compare_pv(
        jds, pv_queried, pv_reference,
        label=label, x_tick=x_tick,
        cols=[1], comp_labels=["dTCL/dt"], units=units,
        show_overview=show_overview,
    )


# ---------------------------------------------------------------------------
# Shared plot implementation
# ---------------------------------------------------------------------------

def _plot_compare_pv(
    jds,
    arr_a,
    arr_b,
    label:         str,
    x_tick:        float,
    cols:          list[int],
    comp_labels:   list[str],
    units:         str,
    label_a:       str  = "Queried",
    label_b:       str  = "Data",
    show_overview: bool = False,
) -> None:
    t_years = jd_to_yr(jds)
    n_comp  = len(cols)
    height  = max(4, 3 * n_comp)

    if show_overview:
        fig_v, axs_v = plt.subplots(n_comp, 1, figsize=(13, height), sharex=True,
                                    gridspec_kw={"hspace": 0.07})
        axs_v = np.atleast_1d(axs_v)
        fig_v.suptitle(f"{label} - Values [{units}]", fontsize=13)
        for i, (col, lbl) in enumerate(zip(cols, comp_labels)):
            axs_v[i].plot(t_years, arr_a[:, col], alpha=0.6, label=label_a)
            axs_v[i].plot(t_years, arr_b[:, col], alpha=0.4, label=label_b)
            axs_v[i].set_ylabel(f"{lbl} [{units}]")
            axs_v[i].grid(True)
            axs_v[i].xaxis.set_major_locator(plt.MultipleLocator(x_tick))
            if i == 0:
                axs_v[i].legend(fontsize=8, loc="upper right")
        axs_v[-1].set_xlabel("Year")
        plt.show()

    fig, axs = plt.subplots(n_comp, 1, figsize=(13, height), sharex=True,
                            gridspec_kw={"hspace": 0.07})
    axs = np.atleast_1d(axs)
    fig.suptitle(f"{label} - Difference [{units}]", fontsize=13)

    for i, (col, lbl) in enumerate(zip(cols, comp_labels)):
        diff = arr_a[:, col] - arr_b[:, col]
        axs[i].plot(t_years, diff, color="orange", marker=".", linestyle="", alpha=0.5)
        axs[i].set_ylabel(f"Δ{lbl} [{units}]")
        axs[i].grid(True)
        axs[i].xaxis.set_major_locator(plt.MultipleLocator(x_tick))

    axs[-1].set_xlabel("Year")
    plt.show()
