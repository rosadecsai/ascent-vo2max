"""Additional field analyses requested in the first revision (Sports Engineering, R1).

For every selected FitRec run (results/fitrec_selected.pkl, written by fitrec_run.py) this script
computes, in one pass:

  decomposition of the v2 revision (one change at a time, all with no gait labels):
    T_road   v1 + HR lag fixed at 55 s, constant road prior
    D_road   v1 + tighter drift prior only (free lag), constant road prior
    v2_road  v2 with a constant road prior (lag fixed + tighter drift prior; no climbing prior)
    v2_trail v2 with a constant trail prior (the same prior on flat and hilly runs)
    v2       v2 as frozen (lag fixed + tighter drift prior + climbing-based surface prior)
  sensitivity to the heart-rate anchors (classic, HR-GAP and v2):
    HRrest 45 and 65 bpm instead of 55; HRmax -10 and +10 bpm around the dataset-derived value.

None of these runs changes the frozen v2 configuration; they are post hoc analyses on all runners
(development and test halves), reported as such.

    python3 fitrec_revision.py          # writes results/fitrec_revision.pkl
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
ANCHORS = {"rest45": (45.0, 0.0), "rest65": (65.0, 0.0), "maxm10": (HR_REST, -10.0), "maxp10": (HR_REST, +10.0)}


def job(rows):
    fr = FitRec()
    out = []
    for r in rows:
        o = dict(id=r["id"], userId=r["userId"])
        try:
            p = fr.prep(r)
            ph = fr.prep(r, walk_heuristic=True)
            hm = float(r["hr_max_u"])
            a = es.ascent_v2(p, HR_REST, hm, "auto")
            o["v2"] = a.get("vo2max", np.nan)
            o["v2_sd"] = a.get("sd", np.nan)
            o["v2_drift"] = a.get("drift", np.nan)
            o["v2_h"] = a.get("h_prior", np.nan)
            o["v2_road"] = es.ascent_v2(p, HR_REST, hm, "road").get("vo2max", np.nan)
            o["v2_trail"] = es.ascent_v2(p, HR_REST, hm, "trail").get("vo2max", np.nan)
            o["T_road"] = es.ascent(p, HR_REST, hm, "road", es.AscentConfig(tau_grid=(55.0,))).get("vo2max", np.nan)
            o["D_road"] = es.ascent(p, HR_REST, hm, "road",
                                    es.AscentConfig(drift_prior=(0.05, 0.03))).get("vo2max", np.nan)
            for nm, (rest, dmax) in ANCHORS.items():
                o["v2_" + nm] = es.ascent_v2(p, rest, hm + dmax, "auto").get("vo2max", np.nan)
                o["G_flat_" + nm] = es.reconstruction(ph, rest, hm + dmax, "flat").get("vo2max", np.nan)
                o["G_trail_hrgap_" + nm] = es.reconstruction(ph, rest, hm + dmax, "trail_hrgap").get("vo2max", np.nan)
        except Exception as e:  # keep going
            o["err_rev"] = str(e)[:60]
        out.append(o)
    return out


if __name__ == "__main__":
    sel = pd.read_pickle(os.path.join(RES, "fitrec_selected.pkl"))
    chunks = [g.to_dict("records") for _, g in sel.groupby(np.arange(len(sel)) % 40)]
    t0 = time.time()
    rows = []
    with Pool(2) as pool:
        for k, out in enumerate(pool.imap_unordered(job, chunks)):
            rows.extend(out)
            print(f"chunk {k + 1}/40, {len(rows)} runs, {time.time() - t0:.0f} s", flush=True)
    res = pd.DataFrame(rows)
    res.to_pickle(os.path.join(RES, "fitrec_revision.pkl"))
    print("done", len(res), f"{time.time() - t0:.0f} s")
