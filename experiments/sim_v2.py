"""ASCENT v2 on the simulation: replays the sessions of the main comparison (same seeds) and of
the structural-mismatch experiment, and adds v2 with the known surface class and with 'auto'.

    python3 sim_v2.py main 300      -> results/main_v2.pkl  (merged with main.pkl on seed/kind/race)
    python3 sim_v2.py mismatch 100  -> results/mismatch_v2.pkl
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sm, es, SURF, KINDS  # noqa: E402
from run_sim import WORLDS, SEED0  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def v2_row(s, r, kind):
    p = es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)
    o = {}
    for tag, surf in (("ASCENT2", SURF[kind]), ("ASCENT2auto", "auto")):
        a = es.ascent_v2(p, r.hr_rest, r.hr_max, surf)
        o[tag] = a.get("vo2max", np.nan)
        o[tag + "_sd"] = a.get("sd", np.nan)
        o[tag + "_n"] = a.get("n", 0)
        o[tag + "_h"] = a.get("h_prior", np.nan)
    return o


def job_main(seed):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng)
    rows = []
    for kind in KINDS:
        for race in (False, True):
            s = sm.random_session(r, kind, race, rng)
            o = v2_row(s, r, kind)
            o.update(seed=seed, kind=kind, race=race)
            rows.append(o)
    return rows


def job_mismatch(seed):
    rows = []
    for wname, opts in WORLDS.items():
        rng = np.random.default_rng(seed)
        r = sm.random_runner(rng)
        for kind in ("road_flat", "trail_runnable", "mountain", "skyrun"):
            for race in (False, True):
                c = sm.make_course(kind, rng)
                lo, hi = sm.INTENSITY[kind][1 if race else 0]
                s = sm.simulate_session(r, c, rng.uniform(lo, hi), race, rng, opts=opts)
                o = v2_row(s, r, kind)
                o.update(seed=seed, world=wname, kind=kind, race=race)
                rows.append(o)
    return rows


if __name__ == "__main__":
    which, n = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    if which == "main":
        tasks, fn, base, keys = list(range(SEED0["main"], SEED0["main"] + n)), job_main, "main.pkl", ["seed", "kind", "race"]
    else:
        tasks, fn, base, keys = list(range(60_000, 60_000 + n)), job_mismatch, "mismatch.pkl", ["seed", "world", "kind", "race"]
    with Pool(2) as pool:
        res = pool.map(fn, tasks, chunksize=2)
    df = pd.DataFrame([r for rr in res for r in rr])
    old = pd.read_pickle(os.path.join(RES, base))
    out = old.merge(df, on=keys, how="left")
    assert len(out) == len(old)
    out.to_pickle(os.path.join(RES, base.replace(".pkl", "_v2.pkl")))
    print(f"{which}: {len(out)} rows in {time.time() - t0:.0f} s")
