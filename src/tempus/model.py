# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/model.py
# Includes the main time offset model functions from post-Newtonian theory, and calls the integrator.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Sequence, Callable, Literal
from .integrate import PreparedGrid, get_integrator

C = 299_792_458.0  # m/s
L_B = np.float64(1.550519768e-8)
L_G = np.float64(6.969290134e-10)
TDB0_S = np.float64(-6.55e-5)  # TDB0 [s], IAU 2006 Res. B3
DAY_S = np.float64(86400.0)
JD_J2000 = 2451545.0

# IAU-fixed reference epoch for the TCB/TDB/TCG secular drift:
# 1977 January 1.0 TAI = JD 2443144.5003725 TT
_T0_INT  = 2443144
_T0_FRAC = np.float64(0.5003725)

IntegratorName = Literal["trapz", "am4", "am8", "am10", "am12", "amk", "rk4"]
GMProvider = Callable[[object], float]


@dataclass(frozen=True)
class TimeScalar:
    """A single Julian Date split into integer and fractional parts (0 <= frac_part < 1).

    Keeps the integer and fractional parts separate to avoid cancellation loss
    when combining a small offset with a large integer JD (~2.45 × 10^6).
    Arithmetic preserves the split invariant via carry propagation.

    Unpacking still works for backward compatibility:  i, f = some_time_scalar
    """
    int_part:  int
    frac_part: np.float64

    def __iter__(self):
        yield self.int_part
        yield self.frac_part

    def __add__(self, days: float) -> TimeScalar:
        frac = self.frac_part + np.float64(days)
        carry = int(np.floor(frac))
        return TimeScalar(self.int_part + carry, frac - np.float64(carry))

    def __radd__(self, days: float) -> TimeScalar:
        return self.__add__(days)

    def __sub__(self, days: float) -> TimeScalar:
        return self.__add__(-days)

    @classmethod
    def from_float(cls, jd: float) -> TimeScalar:
        """Split a scalar Julian Date into a TimeScalar."""
        jd_f64 = np.float64(jd)
        i = int(np.floor(jd_f64))
        return cls(i, jd_f64 - np.float64(i))

    def cli_args(self) -> list[str]:
        """[int_str, frac_str] ready for subprocess CLI calls."""
        return [str(self.int_part), str(self.frac_part)]


@dataclass(frozen=True)
class TimeGrid:
    jd_int: np.ndarray
    jd_frac: np.ndarray

    def __add__(self, days) -> TimeGrid:
        """Add days (scalar or per-point array) with carry, like TimeScalar."""
        frac = self.jd_frac + np.asarray(days, dtype=np.float64)
        carry = np.floor(frac).astype(np.int64)
        return TimeGrid(self.jd_int + carry, frac - carry.astype(np.float64))

    def et(self) -> np.ndarray:
        """Seconds past J2000 for each grid point."""
        return jd_parts_to_et(self.jd_int, self.jd_frac)

    @classmethod
    def from_start_end(cls, start: TimeScalar, end: TimeScalar, step: float) -> TimeGrid:
        """Build a grid from two TimeScalar endpoints and a step size in days."""
        return jd_grid(start.int_part, start.frac_part, end.int_part, end.frac_part, step)


def jd_split(jd):
    """Split a Julian Date into integer and fractional parts with 0 <= frac < 1.

    For a scalar input returns a TimeScalar (supports unpacking: i, f = jd_split(jd)).
    For an array input returns (np.ndarray[int64], np.ndarray[float64]).
    """
    if np.ndim(jd) == 0:
        return TimeScalar.from_float(jd)
    jd_f64 = np.asarray(jd, dtype=np.float64)
    jd_int = np.floor(jd_f64).astype(np.int64)
    return jd_int, jd_f64 - jd_int.astype(np.float64)


def jd_parts_to_et(jd_int: np.ndarray, jd_frac: np.ndarray) -> np.ndarray:
    """
    Convert split JD arrays to ET seconds past J2000 (float64).
    """
    ji = np.asarray(jd_int, dtype=np.int64).reshape(-1).astype(np.longdouble)
    jf = np.asarray(jd_frac, dtype=np.float64).reshape(-1).astype(np.longdouble)
    et = ((ji - np.longdouble(JD_J2000)) + jf) * np.longdouble(DAY_S)
    return np.asarray(et, dtype=np.float64)


def jd_grid(
    jd_start_int: int,   jd_start_frac: np.float64,
    jd_end_int:   int,   jd_end_frac:   np.float64,
    jd_step: float,
) -> TimeGrid:
    """
    Build a JD grid split as jd_int + jd_frac, using a running-carry loop.

    Both the start and end epochs are supplied pre-split (int + frac) to avoid
    cancellation loss from adding a small fractional offset to a large integer
    JD (~2451545).  The loop carries overflow from frac into int at every step,
    so precision is independent of the total number of points.

    The grid is left-closed and right-clipped: every point <= jd_end.
    """
    if jd_step <= 0.0:
        raise ValueError("jd_step must be > 0.")
    if (jd_end_int, jd_end_frac) < (jd_start_int, jd_start_frac):
        raise ValueError("jd_end must be >= jd_start.")

    ints:  list[int]      = []
    fracs: list[np.float64] = []

    cur_int  = int(jd_start_int)
    cur_frac = np.float64(jd_start_frac)
    step     = np.float64(jd_step)

    while (cur_int, cur_frac) <= (jd_end_int, np.float64(jd_end_frac)):
        ints.append(cur_int)
        fracs.append(cur_frac)
        cur_frac += step
        if cur_frac >= np.float64(1.0):
            carry     = int(cur_frac)
            cur_int  += carry
            cur_frac -= np.float64(carry)

    return TimeGrid(
        jd_int=np.array(ints,  dtype=np.int64),
        jd_frac=np.array(fracs, dtype=np.float64),
    )


def jd_tcb_to_tdb(grid: TimeGrid, *, l_b: float = L_B, tdb0_s: float = TDB0_S) -> TimeGrid:
    """
    Convert a TCB grid to TDB via the IAU 2006 Res. B3 defining relation:
        JD_TDB = JD_TCB - L_B*(JD_TCB - T0) + TDB0/86400
    Each epoch converts independently, so nothing accumulates along the grid.
    """
    # Recombining after the split subtraction of T0 is safe: |d_t0| is at most
    # ~1e5-1e6 days (float64 ~6e-11 days), and the L_B factor suppresses that
    # rounding to ~1e-18 days in the correction.
    d_t0 = (grid.jd_int - _T0_INT).astype(np.float64) + (grid.jd_frac - _T0_FRAC)
    return grid + (-l_b * d_t0 + tdb0_s / DAY_S)


def jd_tdb_to_tcb(grid: TimeGrid, *, l_b: float = L_B, tdb0_s: float = TDB0_S) -> TimeGrid:
    """
    Convert a TDB grid to TCB — exact inverse of jd_tcb_to_tdb (the defining
    relation is linear in the epoch), up to one ULP of jd_frac per point:
        JD_TCB = JD_TDB + (L_B*(JD_TDB - T0) - TDB0/86400) / (1 - L_B)
    A uniform TDB grid maps to a uniform TCB grid with step h/(1-L_B).
    """
    d_t0 = (grid.jd_int - _T0_INT).astype(np.float64) + (grid.jd_frac - _T0_FRAC)
    return grid + (l_b * d_t0 - tdb0_s / DAY_S) / (np.float64(1.0) - l_b)


def relabel_to_tcx(grid: TimeGrid, y) -> TimeGrid:
    """
    Re-label an offset series from its native timescale t onto the local
    timescale TCX (Klioner 2026, arXiv:2604.16006 §4).

    Each sample is one event carrying both time labels: at native epoch t_k the
    offset is y_k = t - TCX [seconds], so the same event's TCX label is
    t_k - y_k/86400.  The pair (returned grid, y) is the very same offset
    tabulated against TCX — exact, with no interpolation or iteration.

    Only the grid is returned: the values y are unchanged, which is precisely
    what "re-parameterisation" means — the axis label is relabelled, not the
    function recomputed.  The returned grid is non-uniform by
    O(dTCX/dTCB - 1) ~ 1e-8.
    """
    return grid + (-np.asarray(y, dtype=np.float64) / DAY_S)


def jd_origin_1977(timescale: str) -> TimeScalar:
    """
    The IAU 1977 origin (1977-01-01 00:00:00 TAI at the geocentre) on a TCB or
    TDB axis.  TCB, TCG and TT all read T0 = JD 2443144.5003725 at that event;
    a TDB axis reads the same event at T0 + TDB0/86400 (IAU 2006 Res. B3).
    """
    t0 = TimeGrid(jd_int=np.array([_T0_INT]), jd_frac=np.array([_T0_FRAC]))
    if timescale == "TDB":
        t0 = jd_tcb_to_tdb(t0)
    elif timescale != "TCB":
        raise ValueError(f"The 1977 origin is placed on a TCB or TDB axis, got {timescale!r}.")
    return TimeScalar(int(t0.jd_int[0]), np.float64(t0.jd_frac[0]))


def value_at_epoch(jd_int, jd_frac, y, epoch: TimeScalar, *, n_points: int = 12) -> np.float64:
    """
    Value of a sampled series at one epoch, from a local polynomial (degree
    n_points - 1) through the n_points samples nearest to it.

    Used to zero an integration at an epoch between grid nodes.  On the usual
    18-h grid, 12 points interpolate TCL-TCB to ~3e-14 s, below the 18-h
    integration's own error (~1e-12 s rms for the Moon).  The epoch must lie
    inside the sampled span: there is no extrapolation.
    """
    jd_int = np.asarray(jd_int, dtype=np.int64)
    jd_frac = np.asarray(jd_frac, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.size < n_points:
        raise ValueError(f"Need at least {n_points} samples to interpolate, got {y.size}.")
    at = (int(epoch.int_part), float(epoch.frac_part))
    if not (int(jd_int[0]), float(jd_frac[0])) <= at <= (int(jd_int[-1]), float(jd_frac[-1])):
        raise ValueError(f"Epoch {at[0]}+{at[1]!r} lies outside the sampled span "
                         f"{int(jd_int[0])}+{float(jd_frac[0])!r} .. {int(jd_int[-1])}+{float(jd_frac[-1])!r}.")
    t = (jd_int - at[0]).astype(np.float64) + (jd_frac - at[1])    # days from the epoch
    near = np.argsort(np.abs(t))[:n_points]
    return np.float64(np.polynomial.Polynomial.fit(t[near], y[near], n_points - 1)(0.0))


def jd_tt_to_tdb(
    tt_int: int,
    tt_frac: np.float64,
    eph_path,
) -> TimeScalar:
    """
    Convert a TT epoch (split int + frac JD) to TDB using the time ephemeris
    stored in the file (CALCEPH target 16, "TT - TDB").

    The offset is evaluated directly from the Chebyshev series in the file,
    so it is internally consistent with the planetary positions in that file.

    Requires a TDB-scale ephemeris (eph.gettimescale() == 1).

    Parameters
    ----------
    tt_int, tt_frac : split Julian Date in TT scale.
    eph             : calcephpy.CalcephBin opened on a TDB-scale ephemeris release.

    Returns
    -------
    (tdb_int, tdb_frac) : split JD in TDB. tdb_frac may be slightly negative
                        (same convention as astropy .jd2) - CALCEPH accepts this - but not the CLI.
    """
    from calcephpy import CalcephBin, Constants
    eph = CalcephBin.open(str(eph_path))
    if eph.gettimescale() != 1:
        raise ValueError(
            f"jd_tt_to_tdb requires a TDB-scale ephemeris (gettimescale()=1), "
            f"got {eph.gettimescale()}."
        )
    # CALCEPH target 16: TT - TDB in seconds.
    tt_minus_tdb_sec = np.float64(
        eph.compute_unit(tt_int, tt_frac, 16, 0, Constants.UNIT_SEC)[0]
    )

    # TDB = TT - (TT - TDB)
    tdb_frac = tt_frac - tt_minus_tdb_sec / DAY_S

    if tdb_frac < np.float64(0.0):
        tt_int -= 1
        tdb_frac += np.float64(1.0)

    return TimeScalar(tt_int, tdb_frac)


def jd_tt_to_tcb(
    tt_int: int,
    tt_frac: np.float64,
    eph_path,
) -> TimeScalar:
    """
    Convert a TT epoch (split int + frac JD) to TCB using the time ephemeris
    stored in the file (CALCEPH target 17, "TCG - TCB") together with the
    linear TT-TCG relation derived from L_G (constant LGTCGT) stored in the file.

    Chain:
        TCB - TT = -(TCG - TCB)  -  (TT - TCG)
        TT - TCG = -(L_G / (1 - L_G)) * (JD_TT - T0) * 86400   [IAU Eq. 10.1]

    Iteration: target 17 is parameterised in TCB time. Evaluating at TT introduces
    ~163 ns error. One correction pass (re-evaluate at first-estimate TCB) reduces
    the residual to well below 1 ns.

    Requires a TCB-scale ephemeris (eph.gettimescale() == 2).

    Parameters
    ----------
    tt_int, tt_frac : split Julian Date in TT scale.
    eph             : calcephpy.CalcephBin opened on a TCB-scale ephemeris release.

    Returns
    -------
    (tcb_int, tcb_frac) : split JD in TCB.
    """
    from calcephpy import CalcephBin, Constants
    eph = CalcephBin.open(str(eph_path))
    if eph.gettimescale() != 2:
        raise ValueError(
            f"jd_tt_to_tcb requires a TCB-scale ephemeris (gettimescale()=2), "
            f"got {eph.gettimescale()}."
        )
    try:
        L_G_val = np.float64(eph.getconstant("LGTCGT"))
    except Exception:
        L_G_val = L_G  # fall back to IAU nominal

    # TT - TCG secular drift; split int/frac to avoid cancellation loss.
    dt_int_days  = tt_int - _T0_INT
    dt_frac_days = tt_frac - _T0_FRAC
    dt_sec = np.float64(dt_int_days) * DAY_S + dt_frac_days * DAY_S
    tt_minus_tcg_sec = -(L_G_val / (np.float64(1.0) - L_G_val)) * dt_sec

    # Pass 1: evaluate TCG-TCB at TT epoch (~163 ns error)
    tcg_minus_tcb_sec = np.float64(
        eph.compute_unit(tt_int, tt_frac, 17, 0, Constants.UNIT_SEC)[0]
    )
    tcb_minus_tt_sec = -tcg_minus_tcb_sec - tt_minus_tcg_sec
    tcb_frac_rough   = tt_frac + tcb_minus_tt_sec / DAY_S

    # Pass 2: re-evaluate at corrected TCB (residual < 1 ns)
    carry = int(np.floor(tcb_frac_rough))
    tcb1_int  = tt_int + carry
    tcb1_frac = tcb_frac_rough - np.float64(carry)
    tcg_minus_tcb_sec = np.float64(
        eph.compute_unit(tcb1_int, tcb1_frac, 17, 0, Constants.UNIT_SEC)[0]
    )
    tcb_minus_tt_sec = -tcg_minus_tcb_sec - tt_minus_tcg_sec
    tcb_frac = tt_frac + tcb_minus_tt_sec / DAY_S

    if tcb_frac < np.float64(0.0):
        tt_int -= 1
        tcb_frac += np.float64(1.0)

    return TimeScalar(tt_int, tcb_frac)


def _numerical_acceleration(v: np.ndarray, et: np.ndarray) -> np.ndarray:
    """
    Compute a(t) = dv/dt from v(t) with central differences via np.gradient.
    v: (N,3), et: (N,) in seconds.
    """
    a = np.empty_like(v)
    # gradient handles nonuniform et too (uses second order interior)
    for j in range(3):
        a[:, j] = np.gradient(v[:, j], et, edge_order=2)
    return a


def _pairwise_sa(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Compute S_A = sum_{B != A} mu_B / r_BA for all epochs/bodies.

    This implementation is memory-safe for large body sets: it avoids materializing
    the dense (N, B, B, 3) tensor and instead accumulates pairwise contributions
    in O(N*B) working memory.
    """
    N, B, _ = X.shape
    SA = np.zeros((N, B), dtype=X.dtype)
    one = np.array(1.0, dtype=X.dtype)
    inf = np.array(np.inf, dtype=X.dtype)

    for a in range(B - 1):
        # Distances from body a to all bodies with index > a at each epoch.
        dr = X[:, a + 1 :, :] - X[:, a : a + 1, :]  # (N, B-a-1, 3)
        rab = np.linalg.norm(dr, axis=2)             # (N, B-a-1)
        rab = np.where(rab > 0.0, rab, inf)
        inv_r = one / rab

        mu_tail = mu[a + 1 :][None, :]               # (1, B-a-1)
        SA[:, a] += np.sum(mu_tail * inv_r, axis=1)
        SA[:, a + 1 :] += mu[a] * inv_r

    return SA

# Sun's rotation-pole unit vector in ICRS/BCRS (J2000.0), held fixed at the IAU WGCCRE
# values α₀ = 286.13°, δ₀ = 63.87° — Archinal et al. (2018), CeMDA 130, 22 (the 2015
# report). Its precession is negligible over any relevant integration span.
_sun_pole_rad = np.radians([286.13, 63.87])
N_SUN_ICRS: np.ndarray = np.array([
    np.cos(_sun_pole_rad[1]) * np.cos(_sun_pole_rad[0]),
    np.cos(_sun_pole_rad[1]) * np.sin(_sun_pole_rad[0]),
    np.sin(_sun_pole_rad[1]),
], dtype=np.float64)


def _build_context(eph, *,
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    target, center, ext_bodies: Sequence[object],
    gm_provider: GMProvider,
    use_longdouble: bool = False,
    include_sun_j2: bool = False,):
    """
    Build primitives for TCG(TCB) rate at Earth geocenter.

    Returns dict with:
      et (N,)
      xE, vE (N,3), vE2 (N,)
      X, V (N,B,3), V2 (N,B)
      rAE_vec (N,B,3), rAE (N,B)
      UA (N,B) = mu_A/rAE ; U (N,) = sum_A UA
      SA (N,B) = sum_{B!=A} mu_B/rBA  (pairwise among ext bodies)
      Aacc (N,B,3) acceleration of ext bodies
      mu (B,)
    """
    jd_int = np.asarray(jd_int, dtype=np.int64).reshape(-1)
    jd_frac = np.asarray(jd_frac, dtype=np.float64).reshape(-1)
    if jd_int.size != jd_frac.size:
        raise ValueError("jd_int and jd_frac must have the same size.")
    et = jd_parts_to_et(jd_int, jd_frac)
    dtype = np.longdouble if use_longdouble else np.float64

    # TDB-based ephemerides carry TDB-compatible units; the model is written in
    # TCB. TDB->TCB scales lengths and time intervals by the same 1/(1-L_B), so:
    #   position  (L)      -> * scale
    #   velocity  (L/T)    -> unchanged, the two factors cancel
    #   GM        (L^3/T^2)-> * scale
    # The velocity lines below are therefore deliberately unscaled, not an omission.
    scale = 1.0
    ts = eph.timescale()
    if ts == "TDB":
        l_b = np.float64(eph.constants().get("LBTCBD", L_B))
        scale = 1.0 / (1.0 - l_b)

    stE = eph.state_jd(jd_int, jd_frac, target=target, center=center)
    xE = stE[:, :3].astype(dtype, copy=False) * scale
    vE = stE[:, 3:].astype(dtype, copy=False)
    vE2 = np.einsum("ij,ij->i", vE, vE)

    if not ext_bodies:
        raise ValueError("ext_bodies must be non-empty for TCG(TCB) model (need sum over A != E).")

    # External bodies states
    B = len(ext_bodies)
    X = np.empty((et.size, B, 3), dtype=dtype)
    V = np.empty((et.size, B, 3), dtype=dtype)
    mu = np.empty(B, dtype=dtype)

    for k, b in enumerate(ext_bodies):
        stb = eph.state_jd(jd_int, jd_frac, target=b, center=center)
        X[:, k, :] = stb[:, :3].astype(dtype, copy=False) * scale
        V[:, k, :] = stb[:, 3:].astype(dtype, copy=False)
        mu[k] = dtype(gm_provider(b)) * scale

    V2 = np.einsum("nij,nij->ni", V, V)  # (N,B)

    # r_AE = x_A - x_E  (vector and norm)
    rAE_vec = X - xE[:, None, :]                    # (N,B,3)
    rAE = np.linalg.norm(rAE_vec, axis=2)           # (N,B)

    # UA = mu_A / r_AE 
    UA = mu[None, :] / rAE                          # (N,B)
    U = np.sum(UA, axis=1)                          # (N,)

    # Pairwise potential term S_A = sum_{B != A} mu_B / r_BA.
    # Use a streaming pairwise accumulation to avoid O(N*B*B*3) temporary tensors.
    SA = _pairwise_sa(X, mu)

    # Accelerations of ext bodies (numerical dv/dt): gradient over time axis for all
    # bodies and components at once, avoiding a Python loop over B.
    Aacc = np.empty_like(V)
    for j in range(3):
        Aacc[:, :, j] = np.gradient(V[:, :, j].astype(np.float64), et, axis=0).astype(dtype)

    # Sun J2 quadrupole correction (separate from monopole U, added in _rate_o2).
    # DeltaU_J2 = -GM_Sun * J2 * R^2 / r^3 * P2(sin phi)
    # where phi = heliographic latitude of target body.
    # sin phi = (Sun->target) . n_pole / r = -rAE_vec . N_SUN_ICRS / rAE
    #   (rAE_vec = x_Sun - x_target, so Sun->target = -rAE_vec)
    # P2(sin phi) = (3 sin^2(phi) - 1) / 2
    U_j2 = np.zeros(et.size, dtype=dtype)
    if include_sun_j2:
        sun_idx = next((k for k, b in enumerate(ext_bodies) if int(b) == 10), None)
        if sun_idx is not None:
            j2_sun, r_sun_m = eph.sun_j2_constants()
            r_dist = rAE[:, sun_idx]
            sin_phi = -np.einsum("ni,i->n", rAE_vec[:, sun_idx, :], N_SUN_ICRS) / r_dist
            p2 = 0.5 * (3.0 * sin_phi ** 2 - 1.0)
            U_j2 = -mu[sun_idx] * dtype(j2_sun) * dtype(r_sun_m) ** 2 / r_dist ** 3 * p2

    return {
        "et": et,
        "xE": xE, "vE": vE, "vE2": vE2,
        "X": X, "V": V, "V2": V2,
        "rAE_vec": rAE_vec, "rAE": rAE,
        "UA": UA, "U": U, "U_j2": U_j2,
        "SA": SA,
        "Aacc": Aacc,
        "mu": mu,
    }

def _rate_o2(ctx) -> np.ndarray:
    """
    Return (dTCX/dTCB - 1) at O(c^-2): alpha_E/c^2.
    alpha_E = -(1/2 vE^2 + sum_A mu_A/rAE + U_j2).
    """
    alpha = -(0.5 * ctx["vE2"] + ctx["U"] + ctx["U_j2"])
    return alpha / (C * C)

def _rate_o4(ctx) -> np.ndarray:
    """
    Return (dTCX/dTCB - 1) at O(c^-4): beta_E/c^4 using beta_E structure.

    beta_E =
      -1/8 vE^4
      + 1/2 (U)^2
      + sum_A (mu_A/rAE) * (
            4 vE·vA - 3/2 vE^2 - 2 vA^2
          + 1/2 aA·rAE
          + 1/2 ( (vA·rAE)/rAE )^2
          + sum_{B!=A} mu_B/rBA
        )
    """
    vE = ctx["vE"]                 # (N,3)
    vE2 = ctx["vE2"]               # (N,)
    U = ctx["U"]                   # (N,)

    V = ctx["V"]                   # (N,B,3)
    V2 = ctx["V2"]                 # (N,B)
    UA = ctx["UA"]                 # (N,B) = mu_A/rAE
    SA = ctx["SA"]                 # (N,B) = sum_{B!=A} mu_B/rBA
    Aacc = ctx["Aacc"]             # (N,B,3)
    rAE_vec = ctx["rAE_vec"]       # (N,B,3)
    rAE = ctx["rAE"]               # (N,B)

    # vE · vA   -> (N,B)
    vE_dot_vA = np.einsum("nj,nij->ni", vE, V)

    # aA · rAE  -> (N,B)
    aA_dot_rAE = np.einsum("nij,nij->ni", Aacc, rAE_vec)

    # (vA · rAE / rAE)^2 -> (N,B)
    vA_dot_rAE = np.einsum("nij,nij->ni", V, rAE_vec)
    proj2 = (vA_dot_rAE / rAE) ** 2

    # Bracket term per A
    bracket = (
        4.0 * vE_dot_vA
        - 1.5 * vE2[:, None]
        - 2.0 * V2
        + 0.5 * aA_dot_rAE
        + 0.5 * proj2
        + SA
    )

    beta = (
        (-0.125) * (vE2 * vE2)
        + 0.5 * (U * U)
        + np.sum(UA * bracket, axis=1)
    )
    return beta / (C**4)


def Delta_TCX(eph, *,
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    target,center,ext_bodies: Sequence[object],
    gm_provider: GMProvider,
    integrator: IntegratorName = "trapz",
    amk_k: int | None = None,
    use_prepared_grid: bool = True,
    include_o4: bool = False,
    use_longdouble: bool = False,
    include_sun_j2: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (delta, dtcx_dtcb) where:
      - delta [seconds]: Δ(et) with Δ(et0)=0, dΔ/dTCB = dTCX/dTCB - 1,
        so Δ ~ TCX - TCB (up to the order modeled).
      - dtcx_dtcb [dimensionless]: instantaneous dTCX/dTCB on the input grid.

    The public time input is split Julian Date (`jd_int`, `jd_frac`).
    ET seconds past J2000 are derived internally for integration and
    ephemeris backends that do not expose JD-native queries.

    If include_o4=True, includes beta_E/c^4 term.
    """
    jd_int_arr = np.asarray(jd_int, dtype=np.int64).reshape(-1)
    jd_frac_arr = np.asarray(jd_frac, dtype=np.float64).reshape(-1)
    if jd_int_arr.size != jd_frac_arr.size:
        raise ValueError("jd_int and jd_frac must have the same size.")

    jd_int_eval, jd_frac_eval, start_idx = _extend_time_for_am_startup(jd_int_arr, jd_frac_arr, integrator, amk_k)

    ctx = _build_context(
        eph,
        jd_int=jd_int_eval, jd_frac=jd_frac_eval,
        target=target, center=center,
        ext_bodies=ext_bodies,
        gm_provider=gm_provider,
        use_longdouble=use_longdouble,
        include_sun_j2=include_sun_j2,
    )

    rate = _rate_o2(ctx)
    if include_o4:
        rate = rate + _rate_o4(ctx)

    # dTCX/dTCB = 1 + (dTCX/dTCB - 1); capture before negation for the output rate.
    dtcx_dtcb = np.asarray(1.0 + rate, dtype=np.float64)

    rate = -rate  # I want dΔ/dTCB = -(dTCG/dTCB - 1)

    # Chain-rule correction when the ephemeris is TDB-scaled:
    # et_eval is in TDB seconds, but rate = dΔ/dTCB (per TCB second).
    # Multiplying by dTCB/dTDB = 1/(1-L_B) converts the integrand to dΔ/dTDB,
    # matching the integration variable.  Same factor as in Delta_TT exact formula.
    if eph.timescale() == "TDB":
        l_b = np.float64(eph.constants().get("LBTCBD", L_B))
        rate = rate / (np.float64(1.0) - l_b)

    # Use PreparedGrid for fixed-step integrators.
    # Use et_eval (linearly constructed for AM startup) rather than ctx["et"]
    # (JD round-trip) so the uniformity check sees the clean uniform grid.
    et_eval = jd_parts_to_et(jd_int_eval, jd_frac_eval)
    if use_prepared_grid and integrator in ("trapz", "rk4", "am4", "am8", "am10", "am12"):
        grid = PreparedGrid(et_eval)
        delta = grid.integrate(rate, method=integrator)
    else:
        # Fallback: direct call to integrator
        integ = get_integrator(integrator, amk_k=amk_k)
        delta = integ(rate, et_eval)

    if start_idx > 0:
        return delta[start_idx:] - delta[start_idx], dtcx_dtcb[start_idx:]
    return delta, dtcx_dtcb


def per_leg_ext_bodies(target: int, center: int, ext_bodies: Sequence[object]) -> tuple[list[int], list[int]]:
    """
    Split a shared external-body list into the two per-leg lists Delta_TCX_TCY
    actually integrates with: each leg excludes only its own body, and gains
    the other body of the pair as a perturber (if not already present).
    """
    target, center = int(target), int(center)
    ext = [int(b) for b in ext_bodies]
    ext_target = [b for b in ext if b != target] + ([center] if center not in ext else [])
    ext_center = [b for b in ext if b != center] + ([target] if target not in ext else [])
    return ext_target, ext_center


def Delta_TCX_TCY(eph, *,
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    target, center,
    ext_bodies: Sequence[object],
    gm_provider: GMProvider,
    integrator: IntegratorName = "trapz",
    amk_k: int | None = None,
    include_o4: bool = False,
    use_longdouble: bool = False,
    include_sun_j2: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BCRS-subtraction difference between two local coordinate times
    ("native subtraction", Method B in notes/tcg_tcl_presentation.md).

    Returns (delta, rate) where:
      - delta [seconds]: (TC_center - TC_target) as a function of TCB,
        zero at the first epoch (e.g. TCG - TCL for center=Earth, target=Moon).
      - rate [dimensionless]: d(TC_center - TC_target)/dTCB on the grid.

    Two SSB-centered Delta_TCX integrations on the same grid, subtracted.
    Each leg's external-body list excludes only its own body — the other body
    stays in U_ext, which supplies the pair potential and tidal terms.

    The Lorentz-boost / simultaneity term v_center.r_(target-center)/c^2 is
    deliberately NOT applied: the result is parameterised by TCB, not by
    either local timescale. Subtract boost_term() to obtain the
    center-simultaneity (e.g. Kopeikin-style geocentric) description.
    """
    target, center = int(target), int(center)
    if target == center:
        raise ValueError("target and center must be different bodies.")
    ext_target, ext_center = per_leg_ext_bodies(target, center, ext_bodies)

    common = dict(
        jd_int=jd_int, jd_frac=jd_frac, gm_provider=gm_provider,
        integrator=integrator, amk_k=amk_k, include_o4=include_o4,
        use_longdouble=use_longdouble, include_sun_j2=include_sun_j2,
    )
    d_target, r_target = Delta_TCX(eph, target=target, center=0, ext_bodies=ext_target, **common)
    d_center, r_center = Delta_TCX(eph, target=center, center=0, ext_bodies=ext_center, **common)

    # Delta_TCX delta is TCB - TCX, so TC_center - TC_target = d_target - d_center;
    # its rate is dTC_center/dTCB - dTC_target/dTCB.
    return d_target - d_center, r_center - r_target


def boost_term(eph, *,
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    target, center,
) -> np.ndarray:
    """
    Lorentz-boost / simultaneity term v_center . r_(target-center) / c^2
    [seconds] (notes/tcg_tcl_presentation.md §6). Subtracting it from the
    Delta_TCX_TCY delta converts the TCB-parameterised difference to the
    center-simultaneity description (a la Kopeikin). Not applied by TEMPUS.
    """
    state_tc = eph.state_jd(jd_int, jd_frac, int(target), int(center))  # m, m/s
    state_c = eph.state_jd(jd_int, jd_frac, int(center), 0)
    return np.einsum("ij,ij->i", np.atleast_2d(state_c)[:, 3:6], np.atleast_2d(state_tc)[:, :3]) / (C * C)


def Delta_TT(eph, *,
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    target, center, ext_bodies: Sequence[object],
    gm_provider: GMProvider,
    integrator: IntegratorName = "trapz",
    amk_k: int | None = None,
    use_prepared_grid: bool = True,
    exact: bool = False,
    include_o4: bool = True,
    use_longdouble: bool = False,) -> np.ndarray:
    """
    Returns Δ(et) [seconds] with Δ(et0)=0 where
      dΔ = d(TT - TDB)
    for testing. Uses either the approximate form
      d(TT-TDB) = (L_B + alpha/c^2)(1 + L_B - L_G) - L_G + beta/c^4
    or, if exact=True, the exact form
      ((L_B - L_G) + (1 - L_G)(alpha + beta/c^2)/c^2) / (1 - L_B).

    alpha and beta are computed from the same post-Newtonian context
    used for TCG/TCB modeling. If include_o4=False, beta is set to zero.
    """
    jd_int_arr = np.asarray(jd_int, dtype=np.int64).reshape(-1)
    jd_frac_arr = np.asarray(jd_frac, dtype=np.float64).reshape(-1)
    if jd_int_arr.size != jd_frac_arr.size:
        raise ValueError("jd_int and jd_frac must have the same size.")

    jd_int_eval, jd_frac_eval, start_idx = _extend_time_for_am_startup(jd_int_arr, jd_frac_arr, integrator, amk_k)

    ctx = _build_context(
        eph,
        jd_int=jd_int_eval, jd_frac=jd_frac_eval,
        target=target, center=center,
        ext_bodies=ext_bodies,
        gm_provider=gm_provider,
        use_longdouble=use_longdouble,
    )

    rate_o2 = _rate_o2(ctx)
    rate_o4 = _rate_o4(ctx) if include_o4 else 0.0

    if exact:
        rate = ((L_B - L_G) + (1.0 - L_G) * (rate_o2+rate_o4)) / (1.0 - L_B)
    else:
        rate = (L_B + rate_o2) * (1.0 + L_B - L_G) - L_G + rate_o4

    # Use PreparedGrid for fixed-step integrators.
    # Use et_eval (linearly constructed for AM startup) rather than ctx["et"]
    # (JD round-trip) so the uniformity check sees the clean uniform grid.
    et_eval = jd_parts_to_et(jd_int_eval, jd_frac_eval)
    if use_prepared_grid and integrator in ("trapz", "rk4", "am4", "am8", "am10", "am12"):
        grid = PreparedGrid(et_eval)
        delta = grid.integrate(rate, method=integrator)
    else:
        # Fallback: direct call to integrator
        integ = get_integrator(integrator, amk_k=amk_k)
        delta = integ(rate, et_eval)

    if start_idx > 0:
        return delta[start_idx:] - delta[start_idx]
    return delta


# === Helper Functions to smooth over the Integrator startup for like amk multistep integrators ===

def _am_startup_steps(integrator: IntegratorName, amk_k: int | None) -> int:
    """
    Return AM startup history length k for a given integrator.
    """
    if integrator == "am4":
        return 3
    if integrator == "am8":
        return 7
    if integrator == "am10":
        return 9
    if integrator == "am12":
        return 11
    if integrator == "amk" and amk_k is not None:
        return amk_k
    return 0


def _extend_time_for_am_startup(
    jd_int: np.ndarray,
    jd_frac: np.ndarray,
    integrator: IntegratorName,
    amk_k: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    For AM methods, prepend k uniform steps before jd_int[0]/jd_frac[0] so that
    startup transients occur before the original grid start.

    The step size is derived directly from jd_int[1]-jd_int[0] and jd_frac[1]-jd_frac[0]
    to preserve the int/frac split. Each prepended point is built by subtracting one
    step at a time with an explicit carry, avoiding cancellation loss.

    Returns (jd_int_ext, jd_frac_ext, start_index), where start_index == k points to
    the original jd_int[0]/jd_frac[0].
    """
    k = _am_startup_steps(integrator, amk_k)
    if k <= 0 or jd_int.size < 2:
        return jd_int, jd_frac, 0

    # Step size derived from the first two grid points in split form.
    # d_int is exact (integer diff); d_frac has at most one ULP of cancellation.
    d_int = int(jd_int[1]) - int(jd_int[0])
    d_frac = np.float64(jd_frac[1]) - np.float64(jd_frac[0])
    jd_step_days = np.float64(d_int) + d_frac  # e.g. 0.1 or 1.0

    # Walk backwards k steps from the first grid point, carrying explicitly.
    prepend_int  = np.empty(k, dtype=np.int64)
    prepend_frac = np.empty(k, dtype=np.float64)
    cur_int  = int(jd_int[0])
    cur_frac = np.float64(jd_frac[0])
    for i in range(k):
        cur_frac -= jd_step_days
        if cur_frac < np.float64(0.0):
            cur_int  -= 1
            cur_frac += np.float64(1.0)
        elif cur_frac >= np.float64(1.0):
            cur_int  += 1
            cur_frac -= np.float64(1.0)
        # Store in reverse order so prepend_int[0] is the earliest point.
        prepend_int[k - 1 - i]  = cur_int
        prepend_frac[k - 1 - i] = cur_frac

    jd_int_ext  = np.concatenate([prepend_int,  jd_int])
    jd_frac_ext = np.concatenate([prepend_frac, jd_frac])
    return jd_int_ext, jd_frac_ext, k


