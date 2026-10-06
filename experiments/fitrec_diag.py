"""Diagnostics on the steepest FitRec runs (>= 40 m/km): ASCENT with one change at a time.

Every variant is run on the steep runs AND on the runner's flat runs within +-30 days, so that a
variant that shifts all estimates does not masquerade as a terrain effect.  The within-run
consistency variants (climbs only / flats only) are run on the steep runs only.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascent import estimators as es  # noqa: E402
from ascent.fitrec import FitRec  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HR_REST = 55.0
W = 30 * 86400
PAIRED = {
    "base": dict(surface="road", cfg={}),
    "gait_h": dict(surface="road", cfg={}, walk_heuristic=True),
    "no_lag": dict(surface="road", cfg=dict(use_lag=False)),
    "no_drift": dict(surface="road", cfg=dict(use_drift=False)),
    "no_descents": dict(surface="road", cfg=dict(exclude_descents=0.5)),
    "trail_h": dict(surface="trail", cfg={}),       # on the steep run only: the flat run keeps 'road'
}
WITHIN = {
    "climbs_only": dict(surface="road", cfg=dict(grade_window=(0.04, 1.0))),
    "flats_only": dict(surface="road", cfg=dict(grade_window=(-0.03, 0.04))),
}


def job(rows):
    fr = FitRec()
    out = []
    for r in rows:
        p = fr.prep(r)
        ph = fr.prep(r, walk_heuristic=True)
        o = dict(id=r["id"], role=r["role"])
        var = dict(PAIRED)
        if r["role"] == "steep":
            var.update(WITHIN)
        for k, v in var.items():
            surf = v["surface"] if r["role"] == "steep" else "road"
            a = es.ascent(ph if v.get("walk_heuristic") else p, HR_REST, float(r["hr_max_u"]), surf,
                          es.AscentConfig(**v["cfg"]))
            o[k] = a.get("vo2max", np.nan)          # valid estimates only, as in fitrec_run.py
        o["tau"] = es.ascent(p, HR_REST, float(r["hr_max_u"]), "road").get("tau", np.nan)
        out.append(o)
    return out


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(RES, "fitrec_estimates.pkl"))
    steep = df[df.c >= 40].copy()
    steep["role"] = "steep"
    flats = []
    for uid, g in steep.groupby("userId"):
        fl = df[(df.userId == uid) & (df.cls == "flat")]
        for _, r in g.iterrows():
            flats.append(fl[np.abs(fl.start_ts - r.start_ts) <= W])
    flats = pd.concat(flats).drop_duplicates("id").copy()
    flats["role"] = "flat"
    work = pd.concat([steep, flats])
    print(len(steep), "steep runs,", len(flats), "paired flat runs", flush=True)
    chunks = [g.to_dict("records") for _, g in work.groupby(np.arange(len(work)) % 16)]
    t0 = time.time()
    with Pool(2) as pool:
        rows = sum(pool.map(job, chunks), [])
    est = pd.DataFrame(rows)
    out = work.merge(est, on=["id", "role"])
    out.to_pickle(os.path.join(RES, "fitrec_diag_steep.pkl"))
    fl = out[out.role == "flat"]
    rows = []
    for uid, g in out[out.role == "steep"].groupby("userId"):
        f = fl[fl.userId == uid]
        for _, r in g.iterrows():
            near = f[np.abs(f.start_ts - r.start_ts) <= W]
            if len(near) == 0:
                continue
            d = dict(userId=uid, c=r.c, tau=r.tau)
            for v in PAIRED:
                ref = "base" if v == "trail_h" else v
                d[v] = r[v] - near[ref].mean()
            for v in WITHIN:
                d[v] = r[v] - near["base"].mean()
            d["climb_minus_flat_within"] = r["climbs_only"] - r["flats_only"]
            rows.append(d)
    dr = pd.DataFrame(rows)
    dr.to_pickle(os.path.join(RES, "fitrec_diag_gaps.pkl"))
    u = dr.groupby("userId").mean(numeric_only=True)
    for v in list(PAIRED) + list(WITHIN) + ["climb_minus_flat_within"]:
        x = u[v].dropna()
        print(f"{v:24s} median {x.median():+.2f}  trimmed {stats.trim_mean(x, 0.1):+.2f}  n={len(x)}")
    print("tau at cap (steep):", float((out[out.role == 'steep'].tau >= 120).mean()),
          "(flat):", float((out[out.role == 'flat'].tau >= 120).mean()))
    print("done", len(out), f"{time.time() - t0:.0f} s")
