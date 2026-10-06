"""Trail-running session simulator.

Generates a course (altitude profile, surface), a runner (physiology), a pacing strategy,
the resulting 1-Hz physiology (metabolic demand, heart rate) and what a GPS sports watch
would record (GPS distance, barometric altitude, optical HR, cadence, accelerometer-based
terrain proxy).  The data-generating model is deliberately *richer* than the estimators'
internal models (two-component HR kinetics, mixed additive/multiplicative cardiac drift,
curved HR-VO2 relation, individual uphill/downhill/walking economies, technique-limited
descents, GPS path shortening, stops) so that no estimator is evaluated on its own
assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from . import physiology as ph

DX = 2.0  # horizontal resolution of the course grid (m)

SURFACES = ["road", "pista", "trail", "technical", "very_technical"]
ETA = np.array([1.00, 1.03, 1.07, 1.15, 1.25])          # true metabolic terrain factor
VCAP_DOWN = np.array([5.5, 4.8, 4.0, 3.0, 2.2])          # technique limit on descents (m/s)
VCAP_ANY = np.array([7.0, 6.0, 5.0, 4.0, 3.0])           # technique limit anywhere (m/s)
GPS_EPS_MEAN = np.array([0.005, 0.015, 0.035, 0.055, 0.070])  # path shortening by GPS
GPS_EPS_SD = np.array([0.008, 0.012, 0.020, 0.025, 0.030])
GPS_NOISE = np.array([0.15, 0.20, 0.25, 0.30, 0.35])     # m/s, per-second speed noise

COURSE_KINDS = {
    # name: horizontal length, D+ per km, max |grade|, surface probabilities, start altitude,
    #       correlation lengths of the random grade field, single big climb ("hump")?
    "road_flat": dict(L=(8000, 12000), dpk=(0, 4), gmax=0.02,
                      surf=[1, 0, 0, 0, 0], z0=(600, 700), corr=(1500, 300), hump=False),
    "road_hilly": dict(L=(9000, 11000), dpk=(25, 32), gmax=0.09,
                       surf=[1, 0, 0, 0, 0], z0=(850, 950), corr=(2500, 500), hump=False),
    "trail_runnable": dict(L=(12000, 16000), dpk=(18, 26), gmax=0.10,
                           surf=[0, 0.6, 0.35, 0.05, 0], z0=(550, 650), corr=(1500, 300),
                           hump=False),
    "mountain": dict(L=(10000, 15000), dpk=(40, 55), gmax=0.28,
                     surf=[0, 0.25, 0.45, 0.25, 0.05], z0=(1000, 1300), corr=(2000, 400),
                     hump=False),
    "skyrun": dict(L=(16000, 22000), dpk=(80, 95), gmax=0.35,
                   surf=[0, 0.1, 0.4, 0.35, 0.15], z0=(1400, 1700), corr=(2500, 500),
                   hump=True),
}


# ----------------------------------------------------------------------------------------
@dataclass
class Course:
    kind: str
    x: np.ndarray        # horizontal distance (m)
    z: np.ndarray        # altitude (m)
    grade: np.ndarray    # dz/dx
    surface: np.ndarray  # int index into SURFACES
    eta: np.ndarray      # true terrain factor
    eta_hat: np.ndarray  # accelerometer-based terrain proxy (what a watch could sense)
    gps_eps: np.ndarray  # GPS path-shortening fraction
    obstacle: np.ndarray  # bool: river crossing / scramble (speed capped)


def _scaled_grade(rng, n, kind, L, dplus, gmax, corr, hump):
    g = np.zeros(n)
    for cl, w in zip(corr, (1.0, 0.45)):
        r = gaussian_filter1d(rng.normal(size=n), sigma=cl / DX / 2.0, mode="wrap")
        g += w * r / (r.std() + 1e-12)
    if hump:
        base = np.where(np.arange(n) < n // 2, 1.0, -1.0)
        g = 2.2 * base + 0.6 * g
    g -= g.mean()
    if dplus <= 0:
        return np.zeros(n)
    s = 0.05
    for _ in range(60):
        gg = np.clip(s * g, -gmax, gmax)
        gg -= gg.mean()                       # loop course: end at the start altitude
        dp = np.sum(np.maximum(gg, 0)) * DX
        s *= (dplus / max(dp, 1e-6)) ** 0.8
    gg = np.clip(s * g, -gmax, gmax)
    gg -= gg.mean()
    return gaussian_filter1d(gg, sigma=15.0 / DX)


def make_course(kind: str, rng: np.random.Generator) -> Course:
    p = COURSE_KINDS[kind]
    L = rng.uniform(*p["L"])
    n = int(L / DX)
    dplus = rng.uniform(*p["dpk"]) * L / 1000.0
    g = _scaled_grade(rng, n, kind, L, dplus, p["gmax"], p["corr"], p["hump"])
    if kind == "road_flat":
        g = 0.006 * gaussian_filter1d(rng.normal(size=n), sigma=150) / 0.05
        g = np.clip(g - g.mean(), -0.02, 0.02)
    x = np.arange(n) * DX
    z = rng.uniform(*p["z0"]) + np.concatenate([[0.0], np.cumsum(g[:-1]) * DX])

    # surfaces by segments
    surface = np.zeros(n, int)
    eta = np.ones(n)
    eta_hat = np.ones(n)
    gps_eps = np.zeros(n)
    k = 0
    while k < n:
        seg = int(rng.uniform(300, 1200) / DX)
        sl = slice(k, min(n, k + seg))
        s = rng.choice(5, p=np.array(p["surf"]) / np.sum(p["surf"]))
        if np.mean(np.abs(g[sl])) > 0.15 and s < 3 and rng.random() < 0.5:
            s = 3
        surface[sl] = s
        e_true = ETA[s] + (rng.normal(0, 0.02) if s > 0 else 0.0)
        eta[sl] = max(1.0, e_true)
        eta_hat[sl] = 1.0 + 0.7 * (eta[sl][0] - 1.0) + rng.normal(0, 0.02)
        gps_eps[sl] = np.clip(rng.normal(GPS_EPS_MEAN[s], GPS_EPS_SD[s]), -0.02, 0.15)
        k += seg
    obstacle = np.zeros(n, bool)
    if kind in ("mountain", "skyrun"):
        for _ in range(rng.poisson(1.2)):
            j = rng.integers(int(0.1 * n), int(0.9 * n))
            obstacle[j:j + int(15 / DX)] = True
    return Course(kind, x, z, g, surface, eta, eta_hat, gps_eps, obstacle)


# ----------------------------------------------------------------------------------------
@dataclass
class Runner:
    vo2max: float          # sea-level VO2max (ml/kg/min)
    hr_rest: float
    hr_max: float
    econ: float = 1.0      # running economy multiplier (1 = population ACSM cost)
    up_eff: float = 1.0    # multiplier on the *extra* cost of climbing
    down_mult: float = 1.0  # multiplier on the downhill saving
    walk_econ: float = 1.0
    tau_fast: float = 12.0
    tau_slow: float = 70.0
    w_fast: float = 0.35
    drift: float = 0.05    # relative HR drift per hour
    drift_mix: float = 0.5  # 0 = multiplicative, 1 = additive (90 bpm reference)
    ecc_lambda: float = 0.6  # share of the Strava-Minetti gap that shows up in HR on descents
    gamma: float = 1.0     # curvature of the HR-VO2 relation
    walk_grade: float = 0.15
    skill: float = 1.0     # descending/technical skill (speed caps)
    alt_slope: float = ph.ALT_SLOPE
    hr_noise: float = 2.0
    race_offset: float = 0.0  # bpm of race-day HR inflation (arousal, heat, evening); 0 = none

    @property
    def target(self) -> float:
        """Flat-road 'effective' VO2max that any speed-based method can identify."""
        return ph.VO2_REST + (self.vo2max - ph.VO2_REST) / self.econ


def random_runner(rng: np.random.Generator, **over) -> Runner:
    age_hrmax = rng.uniform(165, 195)
    r = Runner(
        vo2max=rng.uniform(40, 65),
        hr_rest=rng.uniform(40, 60),
        hr_max=age_hrmax,
        econ=float(np.clip(rng.normal(1.0, 0.05), 0.88, 1.12)),
        up_eff=float(np.clip(rng.normal(1.0, 0.05), 0.88, 1.12)),
        down_mult=float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.4)),
        walk_econ=float(np.clip(rng.normal(1.0, 0.06), 0.85, 1.15)),
        tau_fast=rng.uniform(8, 20),
        tau_slow=float(np.clip(rng.normal(70, 20), 35, 130)),
        w_fast=rng.uniform(0.25, 0.45),
        drift=rng.uniform(0.02, 0.10),
        drift_mix=rng.uniform(0, 1),
        ecc_lambda=rng.uniform(0.3, 1.0),
        gamma=rng.uniform(0.95, 1.05),
        walk_grade=rng.uniform(0.10, 0.20),
        skill=rng.uniform(0.8, 1.2),
        alt_slope=float(np.clip(rng.normal(ph.ALT_SLOPE, 0.01), 0.04, 0.09)),
    )
    for k, v in over.items():
        setattr(r, k, v)
    return r


# ----------------------------------------------------------------------------------------
@dataclass
class Session:
    kind: str
    race: bool
    t: np.ndarray          # s
    hr: np.ndarray         # recorded HR (bpm)
    dist: np.ndarray       # recorded cumulative GPS (horizontal) distance (m)
    alt: np.ndarray        # recorded barometric altitude (m)
    cad: np.ndarray        # cadence (Strava units: rpm = steps/min/2)
    eta_hat: np.ndarray    # accelerometer terrain proxy (Garmin-like)
    surface: np.ndarray
    truth: dict = field(default_factory=dict)


def _ar1(rng, n, sigma, rho):
    e = rng.normal(0, sigma * np.sqrt(1 - rho**2), n)
    return lfilter([1.0], [1.0, -rho], e)


def _first_order(u, tau, y0):
    a = np.exp(-1.0 / tau)
    y, _ = lfilter([1 - a], [1, -a], u, zi=[a * y0])
    return y


DEFAULT_OPTS = dict(gps_bias=True, gps_noise=True, terrain=True, ecc=True, kinetics=True,
                    drift=True, altitude=True, stops=True, walk=True, econ_mismatch=True,
                    hr_noise=True, baro_noise=True, obstacles=True, arousal=True, curvature=True,
                    caps=True,
                    # structural-mismatch options (functional forms different from ASCENT's)
                    cost_tilt=0.0, ecc_shape="gap", alt_quad=0.0, drift_shape="linear",
                    cadence_overlap=False, race_ramp=False)


def simulate_session(runner: Runner, course: Course, f0: float, race: bool,
                     rng: np.random.Generator, stops: bool = True,
                     sensor_noise: bool = True, opts: dict | None = None) -> Session:
    o = dict(DEFAULT_OPTS)
    o.update(opts or {})
    if not stops:
        o["stops"] = False
    r = Runner(**{k: getattr(runner, k) for k in runner.__dataclass_fields__})
    if not o["ecc"]:
        r.ecc_lambda = 0.0
    if not o["drift"]:
        r.drift = 0.0
    if not o["altitude"]:
        r.alt_slope = 0.0
    if not o["walk"]:
        r.walk_grade = 10.0
    if not o["econ_mismatch"]:
        r.up_eff = r.down_mult = r.walk_econ = 1.0
    if not o["curvature"]:
        r.gamma = 1.0
    if not o["kinetics"]:
        r.tau_fast = r.tau_slow = 1.0
    c = course
    if not o["terrain"] or not o["gps_bias"] or not o["obstacles"]:
        c = Course(course.kind, course.x, course.z, course.grade, course.surface,
                   course.eta if o["terrain"] else np.ones_like(course.eta),
                   course.eta_hat if o["terrain"] else np.ones_like(course.eta_hat),
                   course.gps_eps if o["gps_bias"] else np.zeros_like(course.gps_eps),
                   course.obstacle if o["obstacles"] else np.zeros_like(course.obstacle))
    n = c.x.size
    i = c.grade
    surf = c.surface
    A = ph.altitude_factor(c.z, r.alt_slope)
    if o["alt_quad"]:
        dz = np.maximum(c.z - ph.ALT_REF, 0.0) / 1000.0
        A = np.clip(A - o["alt_quad"] * dz**2, 0.5, 1.0)

    # pacing: target fraction of altitude-adjusted VO2max
    push = rng.uniform(0.0, 0.04)
    fluct = gaussian_filter1d(rng.normal(size=n), sigma=250 / DX)
    fluct = 0.02 * fluct / (fluct.std() + 1e-12)
    t_guess = c.x / 2.6 / 3600.0
    f = f0 + push * np.clip(i / 0.10, 0, 1) + fluct - 0.02 * t_guess
    if race:
        f = f + 0.02 * np.exp(-c.x / 1500.0)
    f = np.clip(f, 0.4, 0.97)

    p_star = (f * r.vo2max * A - ph.VO2_REST) * ph.E_O2 / 60.0       # W/kg target
    c_lvl = ph.C0_RUN
    extra = ph.cost_run(i) - c_lvl
    c_run = r.econ * (c_lvl + np.where(i >= 0, r.up_eff, r.down_mult) * extra)
    c_run = c_run + (c.eta - 1.0) * r.econ * c_lvl
    c_walk = r.econ * r.walk_econ * ph.cost_walk(i) + (c.eta - 1.0) * 1.5 * r.econ * ph.cost_walk(0.0)
    if o["cost_tilt"]:
        tilt = 1.0 + o["cost_tilt"] * np.tanh(i / 0.15)
        c_run, c_walk = c_run * tilt, c_walk * tilt
    s_run = p_star / c_run
    s_walk_need = p_star / c_walk
    walk = (i > r.walk_grade) & (s_walk_need <= 1.9)
    s = np.where(walk, np.minimum(s_walk_need, 1.9), s_run)
    steep = np.clip(1.0 - 2.5 * np.maximum(-i - 0.10, 0.0), 0.35, 1.0)
    cap = np.where(i < 0, VCAP_DOWN[surf] * r.skill * steep, VCAP_ANY[surf] * r.skill)
    if o["caps"]:
        s = np.minimum(s, cap)
    s = np.where(c.obstacle, np.minimum(s, 0.8), s)
    s = np.maximum(s, 0.3)
    cost = np.where(walk, c_walk, c_run)
    power = cost * s                                                   # W/kg actual

    # time mapping (moving time) along the course
    ds = DX * np.sqrt(1.0 + i**2)
    dt = ds / s
    tm = np.concatenate([[0.0], np.cumsum(dt)])[:-1]
    T_move = tm[-1] + dt[-1]

    # stops (aid stations, gates, photos, route finding)
    stop_list = []
    if o["stops"]:
        rate = (1 / 1800.0) if race else (1 / 900.0)
        k = rng.poisson(rate * T_move)
        for _ in range(k):
            stop_list.append((rng.uniform(0.05, 0.95) * T_move,
                              rng.uniform(5, 20) if race else rng.uniform(10, 60)))
    stop_list.sort()
    T = T_move + sum(d for _, d in stop_list)
    tt = np.arange(0.0, np.floor(T))
    # wall-clock -> moving time
    m = tt.copy()
    stopped = np.zeros(tt.size, bool)
    shift = 0.0
    for (tmv, d) in stop_list:
        start = tmv + shift
        inside = (tt >= start) & (tt < start + d)
        stopped |= inside
        after = tt >= start + d
        m[inside] = tmv
        m[after] -= d
        shift += d
    m = np.clip(m, 0, tm[-1])
    idx = np.clip(np.searchsorted(tm, m, side="right") - 1, 0, n - 1)

    gi = i[idx]
    si = np.where(stopped, 0.0, s[idx])
    pi = np.where(stopped, 0.0, power[idx])
    walk_t = walk[idx] & ~stopped
    z_t = c.z[idx]
    A_t = A[idx]
    vo2_dem = ph.VO2_REST + 60.0 * pi / ph.E_O2 + np.where(stopped, 1.0, 0.0)
    run_desc = (~walk_t) & (~stopped) & (gi < 0)
    if o["ecc_shape"] == "linear":
        shape = (0.256 / 0.09) * np.minimum(-gi, 0.30)       # same excess as the gap at -9 %
    else:
        shape = np.maximum(ph.strava_like_factor(gi) - ph.rel_cost_run(gi), 0)
    ecc = np.where(run_desc, r.ecc_lambda * shape * r.econ * c_lvl * si * 60.0 / ph.E_O2, 0.0)
    frac_v = np.clip((vo2_dem + ecc - ph.VO2_REST) / (r.vo2max * A_t - ph.VO2_REST), 0, 1.05)
    frac_hr = np.sign(frac_v) * np.abs(frac_v) ** r.gamma
    hr_ss0 = r.hr_rest + (r.hr_max - r.hr_rest) * frac_hr
    th = tt / 3600.0
    if o["drift_shape"] == "quadratic":
        th = th**2
    hr_ss = hr_ss0 + r.drift * th * ((1 - r.drift_mix) * (hr_ss0 - r.hr_rest) + r.drift_mix * 90.0)
    if race and o["arousal"]:
        off = r.race_offset * (2.0 * tt / max(tt[-1], 1.0) if o["race_ramp"] else 1.0)
        hr_ss = hr_ss + 5.0 * np.exp(-tt / 240.0) + off
    hr_ss = np.minimum(hr_ss, r.hr_max)
    y0 = r.hr_rest + 15.0
    hr_true = (r.w_fast * _first_order(hr_ss, r.tau_fast, y0)
               + (1 - r.w_fast) * _first_order(hr_ss, r.tau_slow, y0))

    # --- sensors -------------------------------------------------------------------------
    surf_t = surf[idx]
    vh_true = si / np.sqrt(1.0 + gi**2)
    eps = c.gps_eps[idx]
    vh_meas = vh_true * (1.0 - eps)
    alt, hr = z_t.copy(), hr_true.copy()
    if sensor_noise and o["gps_noise"]:
        noise = _ar1(rng, tt.size, 1.0, 0.7) * GPS_NOISE[surf_t]
        vh_meas = vh_meas + np.where(stopped, 0.05, 1.0) * noise
    vh_meas = np.maximum(vh_meas, 0.0)
    if sensor_noise and o["baro_noise"]:
        alt = z_t + np.cumsum(rng.normal(0, 0.03, tt.size)) + rng.normal(0, 0.25, tt.size)
        alt = np.round(alt / 0.2) * 0.2
    if sensor_noise and o["hr_noise"]:
        hr = np.round(hr_true + _ar1(rng, tt.size, r.hr_noise, 0.95))
    dist = np.cumsum(vh_meas)
    if o["cadence_overlap"]:
        cad_w = np.clip(60 + rng.normal(0, 6, tt.size), 45, 76)
        cad_r = np.clip(76 + 6 * (si - 2.8) + rng.normal(0, 3, tt.size), 64, 98)
    else:
        cad_w = np.clip(55 + rng.normal(0, 3, tt.size), 40, 68)
        cad_r = np.clip(80 + 6 * (si - 2.8) + rng.normal(0, 1.5, tt.size), 70, 98)
    cad = np.where(stopped, 0.0, np.where(walk_t, cad_w, cad_r))
    truth = dict(vo2max=r.vo2max, target=r.target, v=si, vh=vh_true, grade=gi, z=z_t,
                 walk=walk_t, stopped=stopped, vo2_dem=vo2_dem, hr=hr_true, ecc=ecc,
                 eta=c.eta[idx], f0=f0)
    return Session(c.kind, race, tt, hr, dist, alt, cad, c.eta_hat[idx], surf_t, truth)


INTENSITY = {
    # (training range, race range) of target fraction of VO2max
    "road_flat": ((0.62, 0.78), (0.86, 0.92)),
    "road_hilly": ((0.62, 0.78), (0.84, 0.90)),
    "trail_runnable": ((0.62, 0.76), (0.82, 0.88)),
    "mountain": ((0.62, 0.74), (0.80, 0.87)),
    "skyrun": ((0.58, 0.68), (0.70, 0.78)),
}


def random_session(runner: Runner, kind: str, race: bool, rng: np.random.Generator, **kw):
    course = make_course(kind, rng)
    lo, hi = INTENSITY[kind][1 if race else 0]
    return simulate_session(runner, course, rng.uniform(lo, hi), race, rng, **kw)
