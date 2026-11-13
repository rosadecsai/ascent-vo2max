"""Recording conditions of the field data, replayed in simulation (first revision).

The FitRec workouts have no cadence and were resampled to 500 points by the dataset's authors.
This experiment asks how much those two conditions, by themselves, change the simulated
terrain-associated difference (mountain-trail session minus flat-road session of the same runner).

Every simulated runner does a flat-road, a hilly-road and a mountain-trail training session.
Each session is analysed under three recording conditions:
  full      1-Hz recording with cadence (as in the main simulation)
  nocad     1-Hz recording, no cadence: the reconstructions use the speed-based walking label of
            the field test, ASCENT uses no gait labels; accelerometer factor 1, surface unknown
  n500      as nocad, after keeping 500 equally spaced samples of the recording (gap flag 35 s)
Estimators: the four reconstructions, ASCENT v1 (road prior, as in the field test) and ASCENT v2
('auto' surface prior, as in the field test).

    python3 sim_sampling.py [N]     -> results/sim_sampling.pkl
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sm, es, SURF  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
KINDS = ["road_flat", "road_hilly", "mountain"]
SEED0 = 70_000
VARIANTS = ("flat", "trail", "trail_nodesc", "trail_hrgap")


def estimates(p_recon, p_asc, r, surf_v1, surf_v2):
    o = {}
    for v in VARIANTS:
        o["G_" + v] = es.reconstruction(p_recon, r.hr_rest, r.hr_max, v).get("vo2max", np.nan)
    o["v1"] = es.ascent(p_asc, r.hr_rest, r.hr_max, surf_v1).get("vo2max", np.nan)
    o["v2"] = es.ascent_v2(p_asc, r.hr_rest, r.hr_max, surf_v2).get("vo2max", np.nan)
    return o


def job(seed):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng)
    rows = []
    for kind in KINDS:
        s = sm.random_session(r, kind, False, rng)
        base = dict(seed=seed, kind=kind, target=r.target, duration_min=s.t[-1] / 60.0)
        # full: 1 Hz with cadence, known surface (as in the main simulation)
        p = es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)
        rows.append(dict(base, cond="full", spacing=1.0, **estimates(p, p, r, SURF[kind], SURF[kind])))
        # nocad: 1 Hz, no cadence, unknown surface (field conditions except the sampling)
        ph = es.prep(s.t, s.hr, s.dist, s.alt, None, None, walk_heuristic=True)
        pn = es.prep(s.t, s.hr, s.dist, s.alt, None, None)
        rows.append(dict(base, cond="nocad", spacing=1.0, **estimates(ph, pn, r, "road", "auto")))
        # n500: 500 equally spaced samples
        k = np.unique(np.round(np.linspace(0, s.t.size - 1, 500)).astype(int))
        ph = es.prep(s.t[k], s.hr[k], s.dist[k], s.alt[k], None, None, max_gap=35.0, walk_heuristic=True)
        pn = es.prep(s.t[k], s.hr[k], s.dist[k], s.alt[k], None, None, max_gap=35.0)
        rows.append(dict(base, cond="n500", spacing=float(np.median(np.diff(s.t[k]))),
                         **estimates(ph, pn, r, "road", "auto")))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(job, list(range(SEED0, SEED0 + n)), chunksize=2)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_pickle(os.path.join(RES, "sim_sampling.pkl"))
    est = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "v1", "v2"]
    for e in est:
        df[e + "_err"] = df[e] - df.target
    print(df.groupby(["cond", "kind"])[[e + "_err" for e in est]].mean().round(2).to_string())
    print(df.groupby(["cond", "kind"])[est].apply(lambda x: x.notna().mean() * 100).round(0).to_string())
    print(df.groupby(["cond", "kind"]).spacing.median())
    print(f"done {len(df)} rows in {time.time() - t0:.0f} s")
