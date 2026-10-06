"""Terrain descriptors for every FitRec run (from the smoothed signals used by the estimators)."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascent.fitrec import FitRec
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
fr = FitRec(); d = fr.index
rows = []; t0 = time.time()
for i, r in d.iterrows():
    try:
        p = fr.prep(r)
    except Exception as e:
        rows.append(dict(id=r["id"], err=str(e)[:40])); continue
    mv = p["moving"]; g = p["grade"][mv]
    dz = np.diff(p["zs"])
    rows.append(dict(id=r["id"], dplus_s=float(np.clip(dz, 0, None).sum()), dminus_s=float(-np.clip(dz, None, 0).sum()),
                     dist_s=float(p["dist"][-1]), dur=float(p["t"][-1]), frac_steep=float(np.mean(np.abs(g) > 0.05)) if mv.any() else np.nan,
                     frac_moving=float(mv.mean()), v_mean=float(p["v"][mv].mean()) if mv.any() else np.nan,
                     hr_p99=float(np.percentile(p["hr"], 99)), hr_med=float(np.median(p["hr"][mv])) if mv.any() else np.nan,
                     alt_mean=float(p["zs"].mean()), alt_rough=float(np.std(np.diff(p["alt"] - p["zs"])))))
    if i % 5000 == 0:
        print(i, f"{time.time() - t0:.0f} s", flush=True)
out = d.merge(pd.DataFrame(rows), on="id")
out.to_pickle(os.path.join(RES, "fitrec_desc.pkl"))
print("done", len(out), f"{time.time() - t0:.0f} s")
