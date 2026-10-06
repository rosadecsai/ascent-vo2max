"""ASCENT v2 on the case-study activities: v2 with the known surface class and with 'auto',
terrain gaps with Welch CIs, race-day drops, and the tracked value."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ascent import estimators as es, strava  # noqa: E402
from real_sensitivity import welch  # noqa: E402

RES = os.path.join(ROOT, "results")
HR_REST, HR_MAX = 52.0, 170.0


def main():
    meta = json.load(open(os.path.join(ROOT, "data", "strava_meta.json")))
    base = pd.read_csv(os.path.join(RES, "real_estimates_hrmax170.csv"))
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "strava", "*.json"))):
        p, info = strava.load(fp, meta)
        o = dict(id=int(info["id"]))
        for tag, surf in (("ASCENT2", info["surface"]), ("ASCENT2auto", "auto")):
            a = es.ascent_v2(p, HR_REST, HR_MAX, surf)
            o[tag] = a.get("vo2max", np.nan)
            o[tag + "_sd"] = a.get("sd", np.nan)
            o[tag + "_h"] = a.get("h_prior", np.nan)
        rows.append(o)
    df = base.merge(pd.DataFrame(rows), on="id")
    df.to_csv(os.path.join(RES, "real_estimates_v2.csv"), index=False)
    out = []
    recent = df[df.date >= "2026-06-01"]
    for name, d in (("all", df), ("recent", recent)):
        road = d[(d.surface == "road") & ~d.race]
        trail = d[(d.surface == "trail") & ~d.race]
        races = d[(d.surface == "trail") & d.race]
        for e in ("ASCENT", "ASCENT2", "ASCENT2auto", "G_trail", "G_trail_hrgap"):
            g = welch(trail[e], road[e])
            rr = welch(races[e], road[e])
            sd = pd.concat([road[e], trail[e]]).std()
            out.append(dict(period=name, estimator=e, gap=g[0], lo=g[1], hi=g[2], n_trail=g[4], n_road=g[5],
                            race_gap=rr[0], race_lo=rr[1], race_hi=rr[2], sd_train=sd,
                            mean_road=road[e].mean(), mean_trail=trail[e].mean()))
    fx = pd.DataFrame(out)
    fx.to_csv(os.path.join(RES, "real_effects_v2.csv"), index=False)
    # tracked value (race-aware) with v2
    d0 = pd.Timestamp(df.date.min())
    trk = es.RaceAwareTracker()
    shown = []
    for _, r in df.sort_values("date").iterrows():
        shown.append(trk.update((pd.Timestamp(r.date) - d0).days, r.ASCENT2, r.ASCENT2_sd, race=bool(r.race)))
    df["disp_ASCENT2_race"] = shown
    df.to_csv(os.path.join(RES, "real_estimates_v2.csv"), index=False)
    pd.set_option("display.width", 250)
    print(fx.round(2).to_string())
    print("last tracked v2:", shown[-1], " v1:", df.sort_values("date").disp_ASCENT_race.iloc[-1])


if __name__ == "__main__":
    main()
