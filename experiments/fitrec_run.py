"""Terrain-invariance test on FitRec (Endomondo) runs: estimate VO2max with ASCENT and the
Garmin-like reconstructions for every selected run of every selected runner.

Selection (from results/fitrec_desc.pkl, written by fitrec_describe.py):
  runs >= 25 min, >= 80 % moving, median HR > 100 bpm, sample spacing <= 30 s (the workouts
  were resampled to 500 points, so long runs are coarse); plausible altitude: mean within -100..3500 m,
  climbing <= 150 m/km, |D+ - D-| <= 50 % of D+ + D-, altitude roughness <= 5 m (some devices
  record altitude garbage, which inflated the steep bin for every estimator);
  terrain class from climbing per km on the smoothed altitude: flat <= 8, hilly >= 25 m/km;
  runners with >= 5 flat and >= 5 hilly runs; per runner at most 30 flat, 60 hilly, 20 mid runs
  (the most recent ones, so that flat and hilly runs interleave in time).
Garmin-like variants exclude slow steep uphill (walking heuristic); ASCENT uses no gait labels.
HR anchors per runner: HRmax = 95th percentile of the per-run 99th-percentile HR over all the
runner's runs (clipped to 150-210), HRrest = 55 bpm for everybody (unknown).  Surface class
'road' and accelerometer factor 1.0 for every run (the terrain of a run is unknown).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ascent import estimators as es  # noqa: E402
from ascent.fitrec import FitRec  # noqa: E402

RES = os.path.join(ROOT, "results")
HR_REST = 55.0
CAP = dict(flat=30, hilly=60, mid=20)


def select():
    d = pd.read_pickle(os.path.join(RES, "fitrec_desc.pkl"))
    d["c"] = d.dplus_s / d.dist_s * 1000
    asym = (d.dplus_s - d.dminus_s).abs() / np.maximum(d.dplus_s + d.dminus_s, 1.0)
    plausible = (d.alt_mean.between(-100, 3500) & (d.c <= 150) & (asym <= 0.5) & (d.alt_rough <= 5))
    ok = d[(d.dur >= 1500) & (d.frac_moving > 0.8) & (d.hr_med > 100) & (d.dur / d.n <= 30) & plausible].copy()
    ok["cls"] = np.where(ok.c <= 8, "flat", np.where(ok.c >= 25, "hilly", "mid"))
    hrmax = d.groupby("userId").hr_p99.quantile(0.95).clip(150, 210).rename("hr_max_u")
    cnt = ok.groupby("userId").cls.value_counts().unstack(fill_value=0)
    users = cnt[(cnt.get("flat", 0) >= 5) & (cnt.get("hilly", 0) >= 5)].index
    ok = ok[ok.userId.isin(users)].sort_values("start_ts", ascending=False)
    ok = ok.groupby(["userId", "cls"]).head(60)
    parts = []
    for cls, cap in CAP.items():
        parts.append(ok[ok.cls == cls].groupby("userId").head(cap))
    sel = pd.concat(parts).merge(hrmax, left_on="userId", right_index=True)
    return sel.sort_values(["userId", "start_ts"]).reset_index(drop=True)


def job(args):
    uid, rows = args
    fr = FitRec()
    out = []
    for r in rows:
        try:
            # Garmin-like: slow steep uphill treated as walking and excluded (a watch has cadence);
            # ASCENT: no gait labels without cadence (ASCENT_gh: with the same heuristic)
            ph = fr.prep(r, walk_heuristic=True)
            p = fr.prep(r)
            o = dict(id=r["id"], userId=uid)
            hm = float(r["hr_max_u"])
            o["ASCENT_gh"] = es.ascent(ph, HR_REST, hm, "road").get("vo2max", np.nan)
            for v in ("flat", "trail", "trail_nodesc", "trail_hrgap"):
                g = es.garmin_like(ph, HR_REST, hm, v)
                o["G_" + v] = g.get("vo2max", np.nan)
                o["G_" + v + "_n"] = g.get("n", 0)
            a = es.ascent(p, HR_REST, hm, "road")
            o["ASCENT"] = a.get("vo2max", np.nan)
            o["ASCENT_any"] = es.value(a)
            o["ASCENT_sd"] = a.get("sd", np.nan)
            o["ASCENT_tau"] = a.get("tau", np.nan)
            o["ASCENT_drift"] = a.get("drift", np.nan)
            o["ASCENT_n"] = a.get("n", 0)
            for k, val in a.get("weight_share", {}).items():
                o["share_" + k] = val
            out.append(o)
        except Exception as e:
            out.append(dict(id=r["id"], userId=uid, err=str(e)[:60]))
    return out


if __name__ == "__main__":
    sel = select()
    sel.to_pickle(os.path.join(RES, "fitrec_selected.pkl"))
    print(len(sel), "runs,", sel.userId.nunique(), "runners;", sel.cls.value_counts().to_dict(), flush=True)
    jobs = [(uid, g.to_dict("records")) for uid, g in sel.groupby("userId")]
    t0 = time.time()
    rows = []
    with Pool(2) as pool:
        for k, out in enumerate(pool.imap_unordered(job, jobs)):
            rows.extend(out)
            if k % 10 == 0:
                print(f"{k + 1}/{len(jobs)} runners, {len(rows)} runs, {time.time() - t0:.0f} s", flush=True)
                pd.DataFrame(rows).to_pickle(os.path.join(RES, "fitrec_estimates_partial.pkl"))
    res = sel.merge(pd.DataFrame(rows), on=["id", "userId"], how="left")
    res.to_pickle(os.path.join(RES, "fitrec_estimates.pkl"))
    print("done", len(res), f"{time.time() - t0:.0f} s")
