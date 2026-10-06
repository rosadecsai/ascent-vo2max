"""Case study: terrain gap (trail minus road training) with confidence intervals, and its
sensitivity to the terrain prior, the HRmax setting, the alpine outlier and the drift term."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ascent import estimators as es, strava  # noqa: E402

RES = os.path.join(ROOT, "results")
HR_REST = 52.0
ALPINE = "19750627336"      # Trevelez, 15 Aug 2026: 5.3 h, mean altitude 2490 m, 61 % hiking


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    t = stats.t.ppf(0.975, dof)
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return d, d - t * se, d + t * se, p, a.size, b.size


def run_all(hr_max):
    meta = json.load(open(os.path.join(ROOT, "data", "strava_meta.json")))
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "strava", "*.json"))):
        p, info = strava.load(fp, meta)
        out = dict(id=info["id"], date=info["date"], surface=info["surface"], race=info["race"],
                   dplus_per_km=info["dplus_per_km"])
        for eh in (1.0, 1.035, 1.07):
            p["eta_hat"][:] = eh if info["surface"] == "trail" else 1.0
            for v in ("trail", "trail_nodesc", "trail_hrgap"):
                out[f"G_{v}@{eh}"] = es.garmin_like(p, HR_REST, hr_max, v)["vo2max"]
        p["eta_hat"][:] = 1.0
        out["G_flat"] = es.garmin_like(p, HR_REST, hr_max, "flat")["vo2max"]
        for h in (1.0, 1.035, 1.07):
            cfg = es.AscentConfig(h_prior={"road": (1.0, 0.03), "trail": (h, 0.06)})
            out[f"ASCENT@{h}"] = es.ascent(p, HR_REST, hr_max, info["surface"], cfg)["vo2max"]
        cfg = es.AscentConfig(use_drift=False)
        out["ASCENT_nodrift"] = es.ascent(p, HR_REST, hr_max, info["surface"], cfg)["vo2max"]
        rows.append(out)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def effects(df, label):
    cols = [c for c in df.columns if c.startswith("G_") or c.startswith("ASCENT")]
    out = []
    for per, per_name in ((None, "all"), ("2026-06-01", "recent")):
        d = df if per is None else df[df.date >= per]
        for excl in (False, True):
            dd = d[d.id != ALPINE] if excl else d
            road = dd[(dd.surface == "road") & ~dd.race]
            trail = dd[(dd.surface == "trail") & ~dd.race]
            races = dd[(dd.surface == "trail") & dd.race]
            for c in cols:
                g = welch(trail[c], road[c])
                rr = welch(races[c], road[c]) if races[c].notna().sum() >= 2 else (np.nan,) * 6
                slope = np.nan
                tr = dd[~dd.race].dropna(subset=[c])
                if len(tr) > 3:
                    slope = 10 * np.polyfit(tr["dplus_per_km"], tr[c], 1)[0]
                out.append(dict(hrmax=label, period=per_name, excl_alpine=excl, estimator=c,
                                gap=g[0], gap_lo=g[1], gap_hi=g[2], gap_p=g[3], n_trail=g[4], n_road=g[5],
                                race_gap=rr[0], race_lo=rr[1], race_hi=rr[2], n_race=rr[4],
                                slope10=slope))
    return pd.DataFrame(out)


if __name__ == "__main__":
    allfx = []
    for hm in (170.0, 184.0):
        df = run_all(hm)
        df.to_csv(os.path.join(RES, f"real_variants_hrmax{int(hm)}.csv"), index=False)
        allfx.append(effects(df, int(hm)))
    fx = pd.concat(allfx)
    fx.to_csv(os.path.join(RES, "real_effects.csv"), index=False)
    pd.set_option("display.width", 250)
    show = fx[(fx.estimator.isin(["G_flat", "G_trail@1.035", "G_trail_nodesc@1.035", "G_trail_hrgap@1.035",
                                  "ASCENT@1.07", "ASCENT@1.035", "ASCENT@1.0", "G_trail@1.07"]))]
    print(show.round(2).to_string())
