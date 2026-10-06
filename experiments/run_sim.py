"""Monte Carlo experiments (main comparison, ablation, sensitivity, longitudinal).

Usage:  python run_sim.py main|ablation|sensitivity|longitudinal|all [N]
Results are written to ../results/<name>.pkl
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sm, es, run_estimators, SURF, KINDS  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RES, exist_ok=True)
SEED0 = {"main": 10_000, "ablation": 20_000, "sensitivity": 30_000, "longitudinal": 40_000}


# ----------------------------------------------------------------------------------------
def job_main(seed):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng)
    rows = []
    for kind in KINDS:
        for race in (False, True):
            s = sm.random_session(r, kind, race, rng)
            o = run_estimators(s, r, kind)
            o.update(seed=seed, kind=kind, race=race, target=r.target, vo2max=r.vo2max,
                     duration_min=s.t[-1] / 60.0,
                     dplus=float(np.sum(np.maximum(np.diff(s.truth["z"]), 0))),
                     walk_frac=float(np.mean(s.truth["walk"])),
                     desc_frac=float(np.mean(s.truth["grade"] < -0.05)),
                     alt_mean=float(np.mean(s.truth["z"])))
            rows.append(o)
    return rows


ABL = {
    "ASCENT (full)": {},
    "no HR kinetics": dict(use_lag=False),
    "no gait model": dict(use_gait=False),
    "no error-budget weights": dict(use_weights=False),
    "no altitude correction": dict(use_altitude=False),
    "no drift term": dict(use_drift=False),
    "no descent model": dict(use_descent_model=False),
    "free h (self-calibrated)": dict(use_h=True),
}


def job_ablation(seed):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng)
    rows = []
    for kind in ["road_flat", "trail_runnable", "mountain", "skyrun"]:
        for race in (False, True):
            s = sm.random_session(r, kind, race, rng)
            p = es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)
            for name, over in ABL.items():
                a = es.ascent(p, r.hr_rest, r.hr_max, SURF[kind], es.AscentConfig(**over))
                rows.append(dict(seed=seed, kind=kind, race=race, variant=name, target=r.target,
                                 est=a.get("vo2max", np.nan), sd=a.get("sd", np.nan)))
    return rows


SENS = {
    "ecc_lambda": [0.0, 0.5, 1.0],
    "gps_scale": [0.0, 1.0, 2.0],
    "terrain_scale": [0.0, 1.0, 2.0],
    "tau_slow": [40.0, 70.0, 110.0],
    "drift": [0.0, 0.05, 0.12],
    "walk_econ": [0.9, 1.0, 1.1],
    "alt_slope": [0.045, 0.063, 0.08],
    "hrmax_err": [-10.0, 0.0, 10.0],
    "hrrest_err": [-5.0, 0.0, 5.0],
}


def _scaled_course(c, gps_scale=1.0, terrain_scale=1.0):
    return sm.Course(c.kind, c.x, c.z, c.grade, c.surface,
                     1.0 + terrain_scale * (c.eta - 1.0), 1.0 + terrain_scale * (c.eta_hat - 1.0),
                     gps_scale * c.gps_eps, c.obstacle)


def job_sens(args):
    seed, factor, level = args
    rng = np.random.default_rng(seed)       # common random numbers across levels
    over = {}
    if factor in ("ecc_lambda", "drift", "walk_econ", "alt_slope"):
        over[factor] = level
    if factor == "tau_slow":
        over.update(tau_slow=level, tau_fast=12.0, w_fast=0.35)
    r = sm.random_runner(rng, **over)
    rows = []
    for kind, race in (("road_flat", False), ("mountain", False), ("mountain", True)):
        c = sm.make_course(kind, rng)
        if factor in ("gps_scale", "terrain_scale"):
            c = _scaled_course(c, **{factor: level})
        lo, hi = sm.INTENSITY[kind][1 if race else 0]
        s = sm.simulate_session(r, c, rng.uniform(lo, hi), race, rng)
        hrm = r.hr_max + (level if factor == "hrmax_err" else 0.0)
        hrr = r.hr_rest + (level if factor == "hrrest_err" else 0.0)
        o = run_estimators(s, r, kind, hr_rest=hrr, hr_max=hrm)
        o.update(seed=seed, factor=factor, level=level, kind=kind, race=race, target=r.target)
        rows.append(o)
    return rows


# ----------------------------------------------------------------------------------------
WEEK = [(1, "road_flat", False), (3, "road_hilly", False), (5, "mountain", False),
        (6, "trail_runnable", False)]


def job_long(args, weeks=16):
    seed, scenario = args
    rng = np.random.default_rng(seed)
    r0 = sm.random_runner(rng)
    race_off = rng.uniform(3.0, 8.0)
    if scenario == "race_effect":
        r0.race_offset = race_off
    gain = rng.uniform(1.5, 3.5)               # true fitness gain over the block
    walk = np.cumsum(rng.normal(0, 0.08, weeks * 7))
    rows = []
    for w in range(weeks):
        sched = list(WEEK)
        if w % 4 == 3:                          # race every 4th week (Sunday)
            sched[-1] = (6, "mountain", True)
        for dow, kind, race in sched:
            day = w * 7 + dow
            r = sm.Runner(**{k: getattr(r0, k) for k in r0.__dataclass_fields__})
            r.vo2max = r0.vo2max + gain * day / (weeks * 7) + walk[day]
            s = sm.random_session(r, kind, race, rng)
            o = run_estimators(s, r, kind, variants=("flat", "trail", "trail_nodesc"))
            o.update(seed=seed, scenario=scenario, week=w, day=day, kind=kind, race=race,
                     target=r.target, vo2max=r.vo2max)
            rows.append(o)
    return rows


def run(name, n):
    t0 = time.time()
    s0 = SEED0[name]
    if name == "main":
        tasks, fn = list(range(s0, s0 + n)), job_main
    elif name == "ablation":
        tasks, fn = list(range(s0, s0 + n)), job_ablation
    elif name == "sensitivity":
        tasks = [(s0 + k, f, lv) for f, levels in SENS.items() for lv in levels for k in range(n)]
        fn = job_sens
    elif name == "longitudinal":
        tasks = [(s0 + k, sc) for sc in ("no_race_effect", "race_effect") for k in range(n)]
        fn = job_long
    with Pool(2) as pool:
        res = pool.map(fn, tasks, chunksize=2)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_pickle(os.path.join(RES, f"{name}.pkl"))
    print(f"{name}: {len(df)} rows in {time.time() - t0:.0f} s", flush=True)
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    sizes = {"main": 300, "ablation": 150, "sensitivity": 80, "longitudinal": 40}
    if len(sys.argv) > 2:
        sizes = {k: int(sys.argv[2]) for k in sizes}
    names = list(sizes) if which == "all" else [which]
    for nm in names:
        run(nm, sizes[nm])


# ----------------------------------------------------------------------------------------
# supplementary: ablation in a 'matched-terrain' world (no unmodelled horizontal cost)
MATCHED_OPTS = dict(gps_bias=False, terrain=False)


def job_ablation_matched(seed):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng)
    rows = []
    for kind in ["road_flat", "trail_runnable", "mountain", "skyrun"]:
        for race in (False, True):
            s = sm.random_session(r, kind, race, rng, opts=MATCHED_OPTS)
            p = es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)
            for name, over in ABL.items():
                o = dict(over)
                o["h_prior"] = {"road": (1.0, 0.03), "trail": (1.0, 0.06)}
                a = es.ascent(p, r.hr_rest, r.hr_max, SURF[kind], es.AscentConfig(**o))
                rows.append(dict(seed=seed, kind=kind, race=race, variant=name, target=r.target,
                                 est=a.get("vo2max", np.nan), sd=a.get("sd", np.nan)))
    return rows


def run_extra(name, n):
    t0 = time.time()
    if name == "ablation_matched":
        tasks, fn = list(range(50_000, 50_000 + n)), job_ablation_matched
    with Pool(2) as pool:
        res = pool.map(fn, tasks, chunksize=2)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_pickle(os.path.join(RES, f"{name}.pkl"))
    print(f"{name}: {len(df)} rows in {time.time() - t0:.0f} s", flush=True)
    return df


# ----------------------------------------------------------------------------------------
# structural mismatch: worlds whose functional forms differ from ASCENT's
WORLDS = {
    "baseline": {},
    "uphill cost +10 %": dict(cost_tilt=0.10),
    "uphill cost -10 %": dict(cost_tilt=-0.10),
    "descent HR excess linear in grade": dict(ecc_shape="linear"),
    "steeper altitude loss (quadratic)": dict(alt_quad=0.012),
    "accelerating drift (quadratic)": dict(drift_shape="quadratic"),
    "overlapping walk/run cadence": dict(cadence_overlap=True),
    "all of the above (+10 %)": dict(cost_tilt=0.10, ecc_shape="linear", alt_quad=0.012,
                                     drift_shape="quadratic", cadence_overlap=True),
}


def job_mismatch(seed):
    rows = []
    for wname, opts in WORLDS.items():
        rng = np.random.default_rng(seed)          # common random numbers across worlds
        r = sm.random_runner(rng)
        for kind in ("road_flat", "trail_runnable", "mountain", "skyrun"):
            for race in (False, True):
                c = sm.make_course(kind, rng)
                lo, hi = sm.INTENSITY[kind][1 if race else 0]
                s = sm.simulate_session(r, c, rng.uniform(lo, hi), race, rng, opts=opts)
                o = run_estimators(s, r, kind)
                o.update(seed=seed, world=wname, kind=kind, race=race, target=r.target)
                rows.append(o)
    return rows


def run_mismatch(n):
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(job_mismatch, list(range(60_000, 60_000 + n)), chunksize=2)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_pickle(os.path.join(RES, "mismatch.pkl"))
    print(f"mismatch: {len(df)} rows in {time.time() - t0:.0f} s", flush=True)
    return df
