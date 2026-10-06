"""Mechanism attribution: bias of each estimator with all mechanisms on, with each one
switched off in turn, and in an idealised world (common random numbers)."""
import sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sm, es, run_estimators, SURF  # noqa

MECH = ["gps_bias", "terrain", "ecc", "kinetics", "drift", "altitude", "stops", "walk",
        "econ_mismatch", "curvature", "obstacles"]
CONDS = ["all"] + ["no_" + m for m in MECH] + ["ideal"]


def opts_for(cond):
    if cond == "all":
        return {}
    if cond == "ideal":
        return {m: False for m in MECH}
    return {cond[3:]: False}


def job(args):
    seed, kind, race = args
    rows = []
    rng0 = np.random.default_rng(seed)
    runner = sm.random_runner(rng0)
    course = sm.make_course(kind, rng0)
    lo, hi = sm.INTENSITY[kind][1 if race else 0]
    f0 = rng0.uniform(lo, hi)
    for cond in CONDS:
        rng = np.random.default_rng(seed + 10_000)
        s = sm.simulate_session(runner, course, f0, race, rng, opts=opts_for(cond))
        o = run_estimators(s, runner, kind)
        o.update(seed=seed, kind=kind, race=race, cond=cond, target=runner.target)
        rows.append(o)
    return rows


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    kinds = ["road_flat", "trail_runnable", "mountain", "skyrun"]
    tasks = [(1000 + k, kind, False) for kind in kinds for k in range(N)]
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(job, tasks, chunksize=4)
    df = pd.DataFrame([r for rr in res for r in rr])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "attribution.pkl")
    df.to_pickle(out)
    for c in ["G_flat", "G_trail", "G_trail_hrgap", "ASCENT"]:
        df["e_" + c] = df[c] - df["target"]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    tab = df.groupby(["kind", "cond"])[["e_G_flat", "e_G_trail", "e_G_trail_hrgap", "e_ASCENT"]].mean().round(2)
    print(tab)
    print("time", round(time.time() - t0, 1))
