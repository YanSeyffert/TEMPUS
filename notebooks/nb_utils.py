# notebooks/nb_utils.py
# Notebook-only utilities: time conversions, plotting helpers, TEMPUS subprocess wrapper.
# Not part of the tempus library.

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from tempus.model import DAY_S, JD_J2000
from tempus.chebycode.validation import (
    plot_compare_p,
    plot_compare_v,
    plot_compare_angle,
    plot_compare_angledot,
    plot_compare_time,
    plot_compare_timedot,
    validate_series,
    print_report,
)


# ── Time conversions ──────────────────────────────────────────────────────────

def jd_to_et(jd: np.ndarray) -> np.ndarray:
    return (jd - JD_J2000) * DAY_S

et_from_jd = jd_to_et


def et_to_jd(et: np.ndarray) -> np.ndarray:
    return et / DAY_S + JD_J2000

jd_from_et = et_to_jd


def jd_to_yr(jd: np.ndarray) -> np.ndarray:
    return (jd - JD_J2000) / 365.25 + 2000.0

yr_from_jd = jd_to_yr


def yr_to_jd(yr: np.ndarray) -> np.ndarray:
    return (yr - 2000.0) * 365.25 + JD_J2000

jd_from_yr = yr_to_jd


# ── Klioner 1977 stitching artifact ───────────────────────────────────────────

JD_1977 = 2443144.5003725  # 1977 Jan 1.0 TAI (TCX-TCB=0 IAU origin), JD in TT


def detect_1977_step(jdi, jdf, y, window_days=10.0, epoch_jd=JD_1977):
    """Estimate the size of Klioner's forward/backward stitching step at 1977.

    Klioner integrated his TCX-TCB series forward and backward from the 1977
    zero-point and stitched the two branches together without matching them
    to better than ~1 ps (confirmed by Klioner directly). Pass a TEMPUS- or
    INPOP-minus-Klioner residual as `y`: since TEMPUS/INPOP are integrated in
    one unbroken pass, the residual isolates Klioner's stitch step from the
    (smooth, shared) secular and periodic terms that would otherwise swamp a
    fit to the raw series. Fits a line to `window_days` of samples immediately
    before and after `epoch_jd` and returns the gap between the two
    extrapolations evaluated at the epoch.
    """
    jd = jdi.astype(np.float64) + jdf
    before = (jd >= epoch_jd - window_days) & (jd < epoch_jd)
    after  = (jd >= epoch_jd) & (jd <= epoch_jd + window_days)
    p_before = np.polyfit(jd[before] - epoch_jd, y[before], 1)
    p_after  = np.polyfit(jd[after]  - epoch_jd, y[after],  1)
    return p_after[-1] - p_before[-1]  # extrapolated values at t=0 (the epoch)


def remove_1977_step(jdi, jdf, y, window_days=10.0, epoch_jd=JD_1977):
    """Return `y` with Klioner's 1977 stitching step removed, plus the jump found.

    `y` should be the same TEMPUS/INPOP-minus-Klioner residual passed to
    `detect_1977_step`. Shifts every sample from `epoch_jd` onward by the
    detected jump so the two integration branches line up continuously.
    """
    jump = detect_1977_step(jdi, jdf, y, window_days, epoch_jd)
    jd = jdi.astype(np.float64) + jdf
    y_fixed = y.copy()
    y_fixed[jd >= epoch_jd] -= jump
    return y_fixed, jump


# ── TEMPUS subprocess wrapper ─────────────────────────────────────────────────

def run_tempus_integration(eph_file, start, end, jd_step, **kwargs):
    """Run ``tempus integrate`` via subprocess and print its output.

    Parameters
    ----------
    eph_file : path or list of paths  — passed to ``--eph-file``
    start, end : TimeScalar           — integration epoch bounds
    jd_step : float                   — step size in days
    **kwargs : forwarded as CLI flags.
        ``True``  → bare flag (e.g. ``out_rate=True`` → ``--out-rate``)
        other     → ``--flag value``  (paths are str-converted automatically)
        ``False`` or ``None`` → flag omitted
        Underscores in key names are replaced by hyphens.
    """
    eph_files = [eph_file] if not isinstance(eph_file, (list, tuple)) else list(eph_file)
    cmd = [
        "tempus", "integrate",
        "--eph-file", *[str(f) for f in eph_files],
        "--jd-start", *start.cli_args(),
        "--jd-end",   *end.cli_args(),
        "--jd-step",  str(jd_step),
    ]
    for k, v in kwargs.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cmd.append(flag)
        elif v is not False and v is not None:
            if isinstance(v, (list, tuple)):
                cmd += [flag, *[str(x) for x in v]]
            else:
                cmd += [flag, str(v)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, end="")
    if res.stderr:
        print(res.stderr, end="")
    if res.returncode != 0:
        raise RuntimeError(f"tempus integrate failed (exit {res.returncode}): {' '.join(cmd)}")
    return res


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_detrended_timescale(et, tlabel, y, ylabel, color, x_tick=10, y_tick=1e-4,
                             paper_mode=False, save_as=None):
    p = np.polyfit(et, y, 1)
    yfit = np.polyval(p, et)
    y_detrended = y - yfit

    fs  = (6.7, 3.5) if paper_mode else (13, 9)
    lfs = 8          if paper_mode else 11
    fig, ax = plt.subplots(figsize=fs)
    ax.set_title(f"Detrended {ylabel}", fontsize=lfs + 1)
    ax.plot(yr_from_jd(jd_from_et(et)), y_detrended, color=color)
    ax.yaxis.set_major_locator(MultipleLocator(y_tick))
    ax.xaxis.set_major_locator(MultipleLocator(x_tick))
    ax.grid(True, which="major", linestyle="--", color="0.7", linewidth=0.8)
    ax.tick_params(labelsize=lfs)
    ax.set_xlabel(tlabel, fontsize=lfs)
    plt.grid(True)
    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, bbox_inches="tight", dpi=300)
    plt.show()

    print("Linear fit slope (s/s):", p[0])
    print("Linear fit slope (ms/day):", p[0] * DAY_S * 1e3)
    print(f"Min and Max of detrended {ylabel} (ms):", np.min(y_detrended) * 1e3, np.max(y_detrended) * 1e3)


def plot_compare_2timescales(et, y1, y1label, y2, y2label, label="TCB-TCX", x_tick=1,
                              overview=True, paper_mode=False, save_as=None,
                              fix_1977_jump=False, jump_window_days=10.0):
    import scipy.stats as stats

    t_years = yr_from_jd(jd_from_et(et))
    fs  = (6.7, 4.0) if paper_mode else (13, 8)
    lfs = 10          if paper_mode else 11

    hs  = 0.14 if paper_mode else 0.07

    if overview:

        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(et, y1)
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(et, y2)
        print(f"{y1label} linear drift: {slope1} s/s, {slope1 * DAY_S * 1e3} ms/day")
        print(f"{y2label} linear drift: {slope2} s/s, {slope2 * DAY_S * 1e3} ms/day")

        fig, axs = plt.subplots(2, 1, figsize=fs, sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": hs})
        axs[0].set_title(f"{label}")
        axs[0].plot(t_years, y1, marker="", color="blue", label=y1label, linestyle=(3, (3, 3)))
        axs[0].plot(t_years, y2, marker="", color="red",  label=y2label, linestyle=(0, (3, 3)))
        axs[0].set_ylabel(f"{label} [s]")
        axs[0].legend(loc="best")
        axs[0].grid(True)

        detrended2 = y2 - (slope2 * et + intercept2)
        detrended2_max = np.max(detrended2)
        detrended2_min = np.min(detrended2)
        print(f"Detrended {y2label} min/max: {detrended2_min:.3e}, {detrended2_max:.3e} s")
        axs[1].plot(t_years, detrended2 * 1e3, marker="", color="red", label=f"{y2label} detrended")
        axs[1].set_ylabel(f"Detrended {y2label} [ms]")
        axs[1].tick_params(labelsize=lfs)
        axs[1].grid(True)
        plt.xlabel("TCB [year]", fontsize=lfs)
        plt.show()

    
    fig, axs = plt.subplots(2, 1, figsize=fs, sharex=True,
                            gridspec_kw={"height_ratios": [2, 1], "hspace": hs})

    difference = y1 - y2
    if fix_1977_jump:
        # One of y1/y2 is a Klioner series stitched from two independent
        # forward/backward integrations at the 1977 origin (see
        # detect_1977_step); that stitch artifact would otherwise dominate
        # both the drift fit and the detrended residual below.
        jd = jd_from_et(et)
        jdi = np.floor(jd).astype(np.int64)
        jdf = jd - jdi
        difference, jump = remove_1977_step(jdi, jdf, difference, window_days=jump_window_days)
        print(f"1977 stitching jump removed: {jump:.3e} s ({jump * 1e12:.2f} ps)")

    slope, intercept, r_value, p_value, std_err = stats.linregress(et, difference)

    difference_ns = difference * 1e9
    axs[0].plot(t_years, difference_ns, marker="", color="orange",
                label=f"Difference {y1label} - {y2label}")
    axs[0].plot(t_years, (slope * et + intercept) * 1e9, color="black",
                label=f"Linear Fit: drift={slope:.3e} s/s")
    axs[0].set_title(f"Difference of {label}: {y1label} vs. {y2label}", fontsize=lfs + 1)
    axs[0].set_ylabel("Difference [ns]", fontsize=lfs)
    
    axs[0].grid(True)
    axs[0].tick_params(labelsize=lfs)
    if not paper_mode:
        # Paper-mode figures skip this: 
        axs[0].legend(loc="lower right", fontsize=lfs)
        axs[0].text(0.02, 0.95,
                    f"Drift slope (s/s): {slope:.3e}\nAverage drift per day (s/day): {slope * DAY_S:.3e}",
                    transform=axs[0].transAxes, verticalalignment="top", fontsize=lfs - 1)

    difference_detrended = difference - (slope * et + intercept)
    axs[1].plot(t_years, difference_detrended * 1e9, marker="", color="orange",
                label=f"Detrended Difference {y1label} - {y2label}")
    axs[1].set_ylabel("Residual [ns]", fontsize=lfs)
    axs[1].tick_params(labelsize=lfs)
    axs[1].grid(True)

    for ax in axs:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_tick))
        #ax.yaxis.get_offset_text().set_size(lfs - 3)

    plt.xlabel("TCB [year]", fontsize=lfs)
    if save_as:
        fig.savefig(save_as, bbox_inches="tight", dpi=300)
    plt.show()

    print("Drift slope (s/s):", slope)
    print("Average drift per day (ps/day):", slope * DAY_S * 1e12)
    print("Average drift per century (ns/cy):", slope * DAY_S * 365.25 * 100 * 1e9)
    periodic_max = np.max(difference_detrended)
    periodic_min = np.min(difference_detrended)
    print("Min and Max of detrended " + label + " (ns): {:.4f}, {:.4f}".format(
        periodic_min * 1e9, periodic_max * 1e9))


def plot_trend_residual(yr, series, trend_scale=1.0, trend_unit="s",
                        resid_scale=1e6, resid_unit="µs",
                        rate_scale=1e6, rate_unit="µs/day", colors=None,
                        xlim=None, title=""):
    """Two-panel plot: overall linear trend (top) and detrended residual (bottom).

    `series` = {label: values [s]} sharing the decimal-year axis `yr`; each
    series' rate is shown in its legend entry. `*_scale` multiply the values (s)
    for display; `rate_scale` multiplies the slope [s/day] for the legend
    (1e6 -> µs/day; 365.25*100*1e9 -> ns/century). `colors` keeps a curve's
    colour across plots. `xlim` = (yr0, yr1) zooms the *displayed* window while
    the rate is still fitted over the full span, so the periodic residual is
    readable and the rate stays exact.
    """
    yr = np.asarray(yr)
    days = (yr - yr[0]) * 365.25   # slope reference (origin-invariant)
    m = np.ones(len(yr), bool) if xlim is None else ((yr >= xlim[0]) & (yr <= xlim[1]))
    colors = colors or [f"C{k}" for k in range(len(series))]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
    for (lab, y), ls, c in zip(series.items(), ("-", "--", ":", "-."), colors):
        s, i = np.polyfit(days, y, 1)   # fit the full span -> true secular rate
        ax0.plot(yr[m], (y * trend_scale)[m], ls, color=c, label=f"{lab}   ({s * rate_scale:+.2f} {rate_unit})")
        ax1.plot(yr[m], ((y - (s * days + i)) * resid_scale)[m], ls, color=c, lw=2)
    if xlim:
        ax0.set_xlim(xlim)
    ax0.set_ylabel(f"offset [{trend_unit}]"); ax0.grid(True); ax0.legend(loc="best")
    if title:
        ax0.set_title(title)
    ax1.set_ylabel(f"residual [{resid_unit}]"); ax1.set_xlabel("year"); ax1.grid(True)
    return fig, (ax0, ax1)
