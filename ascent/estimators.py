"""VO2max estimators: Garmin/Firstbeat-like baselines and the mountain-adapted ASCENT.

All estimators take the same 1-Hz inputs a GPS watch records (time, HR, GPS distance,
barometric altitude, cadence) plus the user's resting and maximal heart rate.

ASCENT = Altitude-, Slope- and Cardiac-kinetics-corrected Estimation with uNcertainty-
weighted Tracking.  Ingredients:
  (1) gait-aware grade cost (Minetti 2002) on the 3-D path speed, with the vertical speed
      taken from the barometer, so that steep climbs are insensitive to GPS shortening;
  (2) first-order heart-rate kinetics: HR is compared with a *lagged* demand, with the
      time constant fitted per session (profile likelihood, prior 55 +- 30 s);
  (3) altitude normalisation of the HR-VO2 relation (Wehrlin & Hallen 2006);
  (4) a cardiac-drift term estimated jointly with VO2max (ridge prior);
  (5) window weights from an explicit error budget of the demand model (grade model,
      terrain, GPS, kinetics); descents get large variances instead of being trusted;
  (6) a session-level uncertainty that feeds a Kalman tracker across sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

from . import physiology as ph

WIN = 30            # s, analysis window
WARMUP = 300        # s ignored at the start by ASCENT
WARMUP_G = 180      # s ignored at the start by the Garmin-like baselines


# ----------------------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------------------
def prep(t, hr, dist, alt, cad=None, eta_hat=None, max_gap=15.0, walk_heuristic=False):
    """Resample to 1 Hz and derive speed, vertical speed, grade, gait, moving flags.

    Gait comes from cadence.  Without cadence, no second is labelled walking unless
    walk_heuristic=True (slow, steep uphill = walking; see the FitRec test for its bias)."""
    t = np.asarray(t, float)
    tg = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1.0)
    interp = lambda y: np.interp(tg, t, np.asarray(y, float))
    hr_g, d_g, z_g = interp(hr), interp(dist), interp(alt)
    cad_g = interp(cad) if cad is not None else np.full(tg.size, np.nan)
    eh = interp(eta_hat) if eta_hat is not None else np.ones(tg.size)
    # gap flag: 1-Hz points lying in a hole of the original sampling
    k = np.clip(np.searchsorted(t, tg), 1, t.size - 1)
    gap = (t[k] - t[k - 1]) > max_gap

    v = np.gradient(d_g)
    v = np.clip(uniform_filter1d(v, 15, mode="nearest"), 0.0, 8.0)
    zs = uniform_filter1d(z_g, 31, mode="nearest")
    vz = uniform_filter1d(np.gradient(zs), 31, mode="nearest")
    # grade over a +-20 s window, requiring at least 15 m of horizontal travel
    h = 20
    dz = np.empty_like(zs); dx = np.empty_like(d_g)
    dz[h:-h] = zs[2 * h:] - zs[:-2 * h]; dx[h:-h] = d_g[2 * h:] - d_g[:-2 * h]
    dz[:h] = dz[h]; dz[-h:] = dz[-h - 1]; dx[:h] = dx[h]; dx[-h:] = dx[-h - 1]
    grade = np.where(dx > 15.0, dz / np.maximum(dx, 1e-6), vz / np.maximum(v, 0.5))
    grade = np.clip(grade, -0.45, 0.45)
    moving = v > 0.5
    has_cad = not (np.all(np.isnan(cad_g)) or np.nanmax(cad_g) <= 0)
    if not has_cad:
        # without cadence, gait cannot be told apart reliably: the speed/grade heuristic classified
        # slow uphill running as walking and biased steep sessions low (FitRec test)
        walk = moving & (v < 1.9) & (grade > 0.08) if walk_heuristic else np.zeros_like(moving)
    else:
        walk = moving & (cad_g > 20) & (cad_g < 70)
        walk |= moving & (cad_g <= 20) & (v < 1.9)       # missing cadence while moving
    return dict(t=tg, hr=hr_g, dist=d_g, alt=z_g, zs=zs, v=v, vz=vz, grade=grade,
                moving=moving, walk=walk, cad=cad_g, eta_hat=eh, gap=gap, has_cadence=has_cad)


def value(res):
    """Session value, falling back to the (flagged) value of an invalid fit."""
    v = res.get("vo2max", np.nan)
    return v if np.isfinite(v) else res.get("vo2max_if_valid", np.nan)


def _windows(n, start):
    s = np.arange(start, n - WIN + 1, WIN)
    return s


def _wmean(a, starts):
    return np.array([a[s:s + WIN].mean() for s in starts])


# ----------------------------------------------------------------------------------------
# Garmin / Firstbeat-like baselines
# ----------------------------------------------------------------------------------------
def garmin_like(p, hr_rest, hr_max, variant="trail", min_windows=20, return_windows=False):
    """Reconstruction of the publicly described Firstbeat/Garmin running method.

    variant 'flat'  : level ACSM speed->VO2, no grade, no altitude correction.
    variant 'trail' : Minetti grade-adjusted speed x accelerometer terrain factor,
                      altitude-corrected (Garmin trail-run VO2max, 2021+ firmware).
    variant 'trail_hrgap' : as 'trail' but with a heart-rate-calibrated (Strava-like) GAP.
    variant 'trail_nodesc': as 'trail' but descending windows (grade < -3 %) are discarded
                      (a 'steel-man' reconstruction that never trusts descents).
    Steady 30-s running windows with HR >= 70 % HRmax; per-window single-point
    extrapolation along %HRR = %VO2R; session value = 10-90 % trimmed mean.
    Grade-adjusted variants apply the grade factor to the 3-D path speed.
    """
    n = p["t"].size
    st = _windows(n, WARMUP_G)
    if st.size == 0:
        return dict(vo2max=np.nan, valid=False, n=0)
    hrw = _wmean(p["hr"], st)
    vw = _wmean(p["v"], st)
    gw = _wmean(p["grade"], st)
    ehw = _wmean(p["eta_hat"], st)
    zw = _wmean(p["zs"], st)
    ok = np.array([p["moving"][s:s + WIN].all() and not p["walk"][s:s + WIN].any()
                   and not p["gap"][s:s + WIN].any()
                   and (p["hr"][s:s + WIN].max() - p["hr"][s:s + WIN].min()) <= 8
                   and p["v"][s:s + WIN].std() <= 0.15 * max(p["v"][s:s + WIN].mean(), 1e-6)
                   for s in st])
    ok &= (hrw >= 0.70 * hr_max) & (vw >= 1.8) & (hrw < hr_max)
    if variant == "flat":
        vo2 = ph.acsm_level_vo2(vw)
        A = np.ones_like(vw)
    else:
        sw = _wmean(p["v"] * np.sqrt(1.0 + p["grade"] ** 2), st)       # path speed
        factor = ph.strava_like_factor(gw) if variant == "trail_hrgap" else ph.rel_cost_run(gw)
        vo2 = ph.VO2_REST + ph.ACSM_NET * 60.0 * sw * factor * ehw
        A = ph.altitude_factor(zw)
        if variant == "trail_nodesc":
            ok &= gw >= -0.03
    vmax_w = ph.vo2max_from_point(vo2, hrw, hr_rest, hr_max) / A
    sel = vmax_w[ok]
    res = dict(n=int(ok.sum()), valid=bool(ok.sum() >= min_windows))
    if sel.size >= 3:
        lo, hi = np.percentile(sel, [10, 90])
        core = sel[(sel >= lo) & (sel <= hi)]
        res["vo2max"] = float(core.mean())
    else:
        res["vo2max"] = np.nan
    if not res["valid"]:
        res["vo2max_if_valid"] = res["vo2max"]
        res["vo2max"] = np.nan
    if return_windows:
        res["windows"] = dict(start=st, ok=ok, vmax=vmax_w, grade=gw, hr=hrw, v=vw)
    return res


# ----------------------------------------------------------------------------------------
# ASCENT
# ----------------------------------------------------------------------------------------
@dataclass
class AscentConfig:
    # prior on the horizontal-cost factor h (terrain roughness x GPS path shortening x
    # level economy), by surface class: (mean, sd).  Literature: +5 % on mildly uneven
    # ground (Voloshina & Ferris 2015); GPS distances under-read by up to 9 % in forest
    # (Gilgen-Ammann et al. 2020).
    h_prior: dict = field(default_factory=lambda: {"road": (1.00, 0.03), "trail": (1.07, 0.06)})
    sd_eta: dict = field(default_factory=lambda: {"road": 0.02, "trail": 0.05})  # within-session
    sd_gps: dict = field(default_factory=lambda: {"road": 0.01, "trail": 0.03})
    sd_grade_run: float = 0.04
    sd_grade_walk: float = 0.06
    descent_slope: float = 1.5       # extra relative sd per unit of descending grade
    ecc_lambda: float = 0.5          # HR-equivalent descent cost: Minetti + 0.5 (Strava - Minetti)
    tau_grid: tuple = (20.0, 35.0, 50.0, 65.0, 80.0, 100.0, 120.0)
    tau_prior: tuple = (55.0, 30.0)  # Hunt et al. 2015: 57.6 +- 23.6 s
    drift_prior: tuple = (0.05, 0.05)  # relative drift per hour
    sd_hr_win: float = 1.2           # bpm, noise of a 30-s HR mean
    sd_int: float = 3.0              # bpm, intercept / linearity uncertainty
    sd_sys_rel: float = 0.025        # session systematic floor (economy, HR anchors)
    sd_walk_econ: float = 0.06       # between-person spread of hiking economy (systematic)
    min_windows: int = 20
    max_sd: float = 4.0
    huber_c: float = 2.0
    v_grid: tuple = (20.0, 90.0, 0.1)
    # ablation switches
    use_lag: bool = True
    use_gait: bool = True
    use_weights: bool = True
    use_altitude: bool = True
    use_drift: bool = True
    use_descent_model: bool = True
    use_h: bool = False              # h estimated per session (tested; increases variance)
    exclude_descents: float = 1.0    # drop windows whose lagged descent share exceeds this
    grade_window: tuple = (-1.0, 1.0)  # diagnostic: keep only windows with mean grade in range
    # surface class 'auto' (v2): h prior interpolated between road and trail by the climbing of
    # the session, c in m/km: road below auto_c[0], trail above auto_c[1]
    auto_c: tuple = (8.0, 40.0)


# ASCENT v2 (frozen after the development half of the FitRec test, October 2026):
#   - HR lag fixed at the literature mean (55 s): the profile over tau was poorly identified on
#     coarsely sampled steep runs and absorbed terrain signal (Section 8.3 of the report);
#   - surface class 'auto' when the terrain is unknown: h prior interpolated by climbing;
#   - tighter drift prior (sd 0.03/h): less session-to-session noise;
#   - no gait labels without cadence (prep default).
V2_CFG = dict(tau_grid=(55.0,), drift_prior=(0.05, 0.03))


def ascent_v2(p, hr_rest, hr_max, surface_class="auto", return_windows=False, **over):
    cfg = AscentConfig(**{**V2_CFG, **over})
    return ascent(p, hr_rest, hr_max, surface_class, cfg, return_windows)


def climb_per_km(p):
    """Climbing per km of the smoothed altitude profile over the whole session (m/km)."""
    dz = np.diff(p["zs"])
    dist = max(float(p["dist"][-1] - p["dist"][0]), 1.0)
    return float(np.clip(dz, 0, None).sum() / dist * 1000.0)


def surface_prior(p, cfg, surface_class):
    """(h0, sd_h, sd_eta, sd_gps) for a surface class; 'auto' interpolates by climbing."""
    if surface_class != "auto":
        h0, sdh = cfg.h_prior[surface_class]
        return h0, sdh, cfg.sd_eta[surface_class], cfg.sd_gps[surface_class]
    w = float(np.clip((climb_per_km(p) - cfg.auto_c[0]) / (cfg.auto_c[1] - cfg.auto_c[0]), 0.0, 1.0))
    (hr, sr), (ht, st_) = cfg.h_prior["road"], cfg.h_prior["trail"]
    return ((1 - w) * hr + w * ht, (1 - w) * sr + w * st_,
            (1 - w) * cfg.sd_eta["road"] + w * cfg.sd_eta["trail"],
            (1 - w) * cfg.sd_gps["road"] + w * cfg.sd_gps["trail"])


def _poly_deriv_ratio(i, walk):
    """C'(i)/C(i) for the Minetti polynomials (for GPS sensitivity)."""
    i = np.clip(i, -0.45, 0.45)
    cr = ph.minetti_run(i)
    dcr = 5 * 155.4 * i**4 - 4 * 30.4 * i**3 - 3 * 43.3 * i**2 + 2 * 46.3 * i + 19.5
    cw = ph.minetti_walk(i)
    dcw = 5 * 280.5 * i**4 - 4 * 58.7 * i**3 - 3 * 76.8 * i**2 + 2 * 51.9 * i + 19.6
    return np.where(walk, dcw / cw, dcr / cr)


def demand(p, cfg: AscentConfig | None = None, surface_class="trail"):
    """Per-second HR-equivalent demand split into a grade part and a level part.

    Returns (Xg, Xl, sd, moving): net VO2 (ml/kg/min) of the Minetti grade cost on the path
    speed, the 'level' component that the horizontal-cost factor h scales
    (net = Xg + (h-1) Xl), and the absolute uncertainty of the demand model.
    """
    cfg = cfg or AscentConfig()
    i = p["grade"]
    walk = p["walk"] if cfg.use_gait else np.zeros_like(p["walk"])
    s = p["v"] * np.sqrt(1.0 + i**2)                    # 3-D path speed
    c_lvl_r, c_lvl_w = ph.C0_RUN, ph.cost_walk(0.0)
    cost = np.where(walk, ph.cost_walk(i), ph.cost_run(i))
    gap = np.zeros_like(i)
    if cfg.use_descent_model:
        gap = np.maximum(ph.strava_like_factor(i) - ph.rel_cost_run(i), 0.0) * c_lvl_r
        gap = np.where(i < 0, gap, 0.0)
        cost = cost + cfg.ecc_lambda * gap
    k = 60.0 / ph.E_O2
    mov = p["moving"]
    Xg = np.where(mov, k * cost * s, 1.0)
    kappa = np.where(walk, 1.5 * c_lvl_w, c_lvl_r)
    Xl = np.where(mov, k * kappa * s, 0.0)
    # error budget (absolute, ml/kg/min)
    sd_g = np.where(walk, cfg.sd_grade_walk, cfg.sd_grade_run)
    if cfg.use_descent_model:
        sd_g = sd_g + cfg.descent_slope * np.maximum(-i - 0.02, 0.0)
    e_grade = sd_g * Xg
    e_desc = 0.5 * k * gap * s                       # half the Minetti-Strava disagreement
    _, _, sd_eta, sd_gps = surface_prior(p, cfg, surface_class)
    e_terr = sd_eta * Xl
    sens = 1.0 - i * (_poly_deriv_ratio(i, walk) + i / (1.0 + i**2))
    e_gps = sd_gps * np.abs(sens) * Xg
    sd = np.where(mov, np.sqrt(e_grade**2 + e_desc**2 + e_terr**2 + e_gps**2), 0.5)
    return Xg, Xl, sd, mov


def _lag(u, tau):
    if tau <= 0:
        return u.copy()
    a = np.exp(-1.0 / tau)
    y, _ = lfilter([1 - a], [1, -a], u, zi=[a * u[0]])
    return y


def ascent(p, hr_rest, hr_max, surface_class="trail", cfg: AscentConfig | None = None,
           return_windows=False):
    cfg = cfg or AscentConfig()
    n = p["t"].size
    st = _windows(n, WARMUP)
    if st.size < 3:
        return dict(vo2max=np.nan, valid=False, n=0)
    Xg, Xl, sd_abs, mov = demand(p, cfg, surface_class)
    A = ph.altitude_factor(p["zs"]) if cfg.use_altitude else np.ones(n)
    hrw = _wmean(p["hr"], st)
    Aw = _wmean(A, st)
    tw = (st + WIN / 2.0) / 3600.0
    movw = _wmean(mov.astype(float), st)
    gapw = np.array([p["gap"][s:s + WIN].any() for s in st])
    rngw = np.array([p["hr"][s:s + WIN].max() - p["hr"][s:s + WIN].min() for s in st])
    frac = (hrw - hr_rest) / (hr_max - hr_rest)
    base_ok = (movw >= 0.9) & ~gapw & (rngw <= 20) & (frac >= 0.40) & (frac <= 0.97)
    y = hrw - hr_rest
    v0, v1, dv = cfg.v_grid
    V = np.arange(v0, v1 + 1e-9, dv)
    h0, sdh, _, _ = surface_prior(p, cfg, surface_class)
    if not cfg.use_h:
        sdh = 1e-6
    taus = cfg.tau_grid if cfg.use_lag else (0.0,)
    best = None
    for tau in taus:
        xg = _wmean(_lag(Xg, tau), st)
        xlv = _wmean(_lag(Xl, tau), st)
        sdw = _wmean(_lag(sd_abs, tau), st)
        x0 = xg + (h0 - 1.0) * xlv
        if tau > 0:
            tot = Xg + (h0 - 1.0) * Xl
            hi = _wmean(_lag(tot, tau * 1.5), st)
            lo = _wmean(_lag(tot, tau / 1.5), st)
            trans = 0.5 * (np.abs(hi - x0) + np.abs(lo - x0))
        else:
            trans = np.zeros_like(x0)
        ok = base_ok & (x0 > 8.0)
        if cfg.grade_window != (-1.0, 1.0):
            gw_ = _wmean(p["grade"], st)
            ok &= (gw_ >= cfg.grade_window[0]) & (gw_ <= cfg.grade_window[1])
        if cfg.exclude_descents < 1.0:
            dsc = ((p["grade"] < -0.03) & mov).astype(float)
            ok &= _wmean(_lag(dsc, tau), st) <= cfg.exclude_descents
        if ok.sum() < cfg.min_windows:
            continue
        rel = np.sqrt(sdw[ok] ** 2 + trans[ok] ** 2) / x0[ok]
        if not cfg.use_weights:
            rel = np.full_like(rel, np.median(rel))
        yo = y[ok]
        var = cfg.sd_hr_win**2 + cfg.sd_int**2 + (yo * rel) ** 2
        fit = _profile_fit(V, xg[ok], xlv[ok], yo, Aw[ok], tw[ok], var, hr_rest, hr_max,
                           (h0, sdh), cfg)
        crit = fit["S"] + (((tau - cfg.tau_prior[0]) / cfg.tau_prior[1]) ** 2 if tau > 0 else 0.0)
        if best is None or crit < best["crit"]:
            best = dict(crit=crit, tau=tau, fit=fit, ok=ok, rel=rel, x0=x0)
    if best is None:
        return dict(vo2max=np.nan, valid=False, n=0)
    f = best["fit"]
    wk_all = _wmean(p["walk"].astype(float), st)[best["ok"]]
    walk_share = float((f["w"] * wk_all).sum() / f["w"].sum())
    sd_sys = np.hypot(cfg.sd_sys_rel * f["V"], walk_share * cfg.sd_walk_econ * (f["V"] - ph.VO2_REST))
    sd = float(np.hypot(f["sd"], sd_sys))
    nwin = int(best["ok"].sum())
    res = dict(vo2max=f["V"], sd=sd, sd_stat=f["sd"], tau=best["tau"], drift=f["delta"],
               h=f["h"], h_prior=h0, n=nwin,
               valid=bool(nwin >= cfg.min_windows and f["sd"] <= cfg.max_sd
                          and v0 + 1 < f["V"] < v1 - 1))
    g = _wmean(p["grade"], st)[best["ok"]]
    wk = _wmean(p["walk"].astype(float), st)[best["ok"]]
    w = f["w"]
    res["weight_share"] = dict(
        climb=float(w[(g >= 0.04)].sum() / w.sum()),
        flat=float(w[(g > -0.03) & (g < 0.04)].sum() / w.sum()),
        descent=float(w[g <= -0.03].sum() / w.sum()),
        walk=float(w[wk > 0.5].sum() / w.sum()))
    if not res["valid"]:
        res["vo2max_if_valid"] = res["vo2max"]
        res["vo2max"] = np.nan
    if return_windows:
        res["windows"] = dict(start=st, ok=best["ok"], x0=best["x0"], hr=hrw, A=Aw, t=tw,
                              grade=_wmean(p["grade"], st), rel=best["rel"], w_ok=w)
    return res


def _S_of_V(V, xg, xl, y, A, t, w, hr_rest, hr_max, hprior, dprior, use_h):
    """Penalised weighted SSE profiled over (h, delta) for each candidate VO2max in V."""
    h0, sdh = hprior
    d0, sdd = dprior
    b = (hr_max - hr_rest) / (V[:, None] * A[None, :] - ph.VO2_REST)          # (nV, nw)
    W = w[None, :]
    u = np.full(V.size, h0 - 1.0)
    dl = np.full(V.size, d0)
    for _ in range(3 if use_h else 1):
        if use_h:
            g = 1.0 + dl[:, None] * t[None, :]
            pg = b * g * xg[None, :]
            q = b * g * xl[None, :]
            u = ((W * q * (y[None, :] - pg)).sum(1) + (h0 - 1.0) / sdh**2) / \
                ((W * q * q).sum(1) + 1.0 / sdh**2)
        a = b * (xg[None, :] + u[:, None] * xl[None, :])
        at = a * t[None, :]
        dl = ((W * at * (y[None, :] - a)).sum(1) + d0 / sdd**2) / ((W * at * at).sum(1) + 1.0 / sdd**2)
    r = y[None, :] - a * (1.0 + dl[:, None] * t[None, :])
    S = (W * r * r).sum(1) + (dl - d0) ** 2 / sdd**2
    if use_h:
        S = S + (u - (h0 - 1.0)) ** 2 / sdh**2
    return S, r, 1.0 + u, dl


def _profile_fit(V, xg, xl, y, A, t, var, hr_rest, hr_max, hprior, cfg):
    """Profile likelihood over VO2max.

    Model per window: y = b(V) * (xg + (h-1) xl) * (1 + delta t),  b = (HRmax-HRrest)/(V A - 3.5),
    Gaussian priors on h (fixed at its prior mean unless cfg.use_h) and on the drift delta,
    Huber-type robust reweighting; coarse-to-fine search over V.
    """
    dprior = cfg.drift_prior if cfg.use_drift else (0.0, 1e-6)
    use_h = cfg.use_h
    Vc = np.arange(V[0], V[-1] + 1e-9, 0.5)
    rob = np.ones_like(y)
    for _ in range(4):
        w = rob / var
        S, r, h, dl = _S_of_V(Vc, xg, xl, y, A, t, w, hr_rest, hr_max, hprior, dprior, use_h)
        k = int(np.argmin(S))
        z = r[k] / np.sqrt(var)
        new = np.minimum(1.0, cfg.huber_c / np.maximum(np.abs(z), 1e-9))
        if np.allclose(new, rob, atol=1e-3):
            break
        rob = new
    w = rob / var
    Vf = np.arange(max(V[0], Vc[k] - 1.0), min(V[-1], Vc[k] + 1.0) + 1e-9, cfg.v_grid[2])
    Sf, rf, hf, dlf = _S_of_V(Vf, xg, xl, y, A, t, w, hr_rest, hr_max, hprior, dprior, use_h)
    kf = int(np.argmin(Sf))
    Smin = float(Sf[kf])
    n_eff = max(len(y) / 2.0, 4.0)
    phi = max(1.0, Smin / (n_eff - 2.0))
    inside = Vc[S - Smin <= phi]
    if inside.size >= 2:
        sd = (inside.max() - inside.min() + 0.5) / 2.0
    else:
        sd = 0.25
    return dict(V=float(Vf[kf]), sd=float(sd), delta=float(dlf[kf]), h=float(hf[kf]), S=Smin,
                w=w)


# ----------------------------------------------------------------------------------------
# across-session tracking
# ----------------------------------------------------------------------------------------
class KalmanTracker:
    """Random-walk fitness state; each session enters with its own uncertainty."""

    def __init__(self, q_per_day=0.05, r_floor=1.0):
        self.q, self.r_floor = q_per_day, r_floor
        self.m = None
        self.P = None
        self.day = None

    def update(self, day, z, sd):
        if z is None or not np.isfinite(z):
            return self.m
        R = sd**2 + self.r_floor
        if self.m is None:
            self.m, self.P, self.day = z, R, day
            return self.m
        self.P += self.q * max(day - self.day, 0)
        self.day = day
        K = self.P / (self.P + R)
        self.m += K * (z - self.m)
        self.P *= (1 - K)
        return self.m


class RaceAwareTracker:
    """Two-state Kalman tracker x = [V, b_race].

    Training sessions observe V; race sessions observe V + b_race, where b_race is a
    person-specific race-day offset (arousal, heat or evening start, fast descents,
    cramps...).  b_race ~ N(0, sd_b^2) a priori and is learned from the athlete's own
    races, so early races barely move V and later races are de-biased.
    """

    def __init__(self, q_per_day=0.05, r_floor=1.0, sd_b=4.0):
        self.q, self.r_floor, self.sd_b = q_per_day, r_floor, sd_b
        self.x = None
        self.P = None
        self.day = None

    @property
    def m(self):
        return None if self.x is None else float(self.x[0])

    @property
    def b(self):
        return None if self.x is None else float(self.x[1])

    def update(self, day, z, sd, race=False):
        if z is None or not np.isfinite(z):
            return self.m
        R = sd**2 + self.r_floor
        H = np.array([1.0, 1.0 if race else 0.0])
        if self.x is None:
            if race:            # no fitness information yet: store nothing until training data
                return None
            self.x = np.array([z, 0.0])
            self.P = np.diag([R, self.sd_b**2])
            self.day = day
            return self.m
        self.P[0, 0] += self.q * max(day - self.day, 0)
        self.day = day
        S = H @ self.P @ H + R
        K = self.P @ H / S
        self.x = self.x + K * (z - H @ self.x)
        self.P = self.P - np.outer(K, H @ self.P)
        return self.m


class EWMADisplay:
    """Constant-gain smoother (our stand-in for the watch's displayed value)."""

    def __init__(self, gain=0.3):
        self.g, self.m = gain, None

    def update(self, day, z, sd=None):
        if z is None or not np.isfinite(z):
            return self.m
        self.m = z if self.m is None else self.m + self.g * (z - self.m)
        return self.m
