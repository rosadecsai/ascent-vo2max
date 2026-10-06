"""Apply the Garmin-like baselines and ASCENT to the athlete's Strava activities.

The ASCENT configuration is the one frozen after the simulation study (AscentConfig()
defaults); nothing is tuned on these data.  HRrest = 52 bpm (athlete: 50-55), HRmax = 170 bpm
(configured on the watch); a sensitivity run uses a data-based HRmax.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ascent import estimators as es, strava  # noqa: E402

RES = os.path.join(ROOT, "results")
HR_REST = 52.0
HR_MAX_SET = 170.0
TRAIL_ETA_HAT = 1.035   # what a Garmin-like accelerometer factor would add on average on trails


def check_frozen():
    """The ASCENT defaults must equal the configuration frozen after the simulation study."""
    import dataclasses
    frozen = json.load(open(os.path.join(RES, "frozen_ascent_config.json")))
    now = json.loads(json.dumps(dataclasses.asdict(es.AscentConfig()), default=str))
    diff = {k: (frozen[k], now.get(k)) for k in frozen if now.get(k) != frozen[k]}
    assert not diff, f"ASCENT defaults differ from the frozen configuration: {diff}"
    extra = sorted(set(now) - set(frozen))
    return extra


def analyse(hr_max, keep_windows=()):
    meta = json.load(open(os.path.join(ROOT, "data", "strava_meta.json")))
    rows, wins = [], {}
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "strava", "*.json"))):
        p, info = strava.load(fp, meta)
        p["eta_hat"][:] = TRAIL_ETA_HAT if info["surface"] == "trail" else 1.0
        out = dict(info)
        for v in ("flat", "trail", "trail_nodesc", "trail_hrgap"):
            g = es.garmin_like(p, HR_REST, hr_max, v, return_windows=info["id"] in keep_windows)
            out["G_" + v] = g.get("vo2max", np.nan)
            out["G_" + v + "_n"] = g.get("n", 0)
            if info["id"] in keep_windows and v == "trail":
                wins[(info["id"], "G_trail")] = g["windows"]
        a = es.ascent(p, HR_REST, hr_max, info["surface"], return_windows=info["id"] in keep_windows)
        out["ASCENT"] = a.get("vo2max", np.nan)
        out["ASCENT_any"] = es.value(a)
        out["ASCENT_sd"] = a.get("sd", np.nan)
        out["ASCENT_tau"] = a.get("tau", np.nan)
        out["ASCENT_drift"] = a.get("drift", np.nan)
        out["ASCENT_n"] = a.get("n", 0)
        for k, val in a.get("weight_share", {}).items():
            out["share_" + k] = val
        if info["id"] in keep_windows:
            wins[(info["id"], "ASCENT")] = a.get("windows")
            wins[(info["id"], "prep")] = p
        rows.append(out)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df, wins


def slope_ci(x, y, n_boot=4000, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    b = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, x.size, x.size)
        if np.ptp(x[k]) > 0:
            bs.append(np.polyfit(x[k], y[k], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return b, lo, hi, x.size


def summarise(df, label):
    lines = [f"== {label} =="]
    for c in ("G_flat", "G_trail", "G_trail_hrgap", "ASCENT"):
        road = df.loc[df.surface == "road", c].dropna()
        trail = df.loc[df.surface == "trail", c].dropna()
        b, lo, hi, n = slope_ci(df["dplus_per_km"].values, df[c].values)
        lines.append(f"{c:14s} road {road.mean():5.1f}+-{road.std():3.1f} (n={road.size:2d})  "
                     f"trail {trail.mean():5.1f}+-{trail.std():3.1f} (n={trail.size:2d})  "
                     f"diff {trail.mean() - road.mean():+5.1f}  slope/10 m/km {10 * b:+5.2f} "
                     f"[{10 * lo:+5.2f},{10 * hi:+5.2f}] n={n}")
    return "\n".join(lines)


def track(df, est_col, sd_col=None, kind="ewma", gain=0.3):
    """Displayed value after each activity (chronological)."""
    d0 = pd.Timestamp(df["date"].min())
    out = []
    if kind == "ewma":
        tr = es.EWMADisplay(gain)
    elif kind == "kalman":
        tr = es.KalmanTracker()
    else:
        tr = es.RaceAwareTracker()
    for _, r in df.iterrows():
        day = (pd.Timestamp(r["date"]) - d0).days
        z = r[est_col]
        sd = r[sd_col] if sd_col else None
        if kind == "race_aware":
            m = tr.update(day, z, sd, race=bool(r["race"]))
        else:
            m = tr.update(day, z, sd)
        out.append(np.nan if m is None else m)
    return np.array(out)


if __name__ == "__main__":
    print("frozen-config check passed; options added later (diagnostic only):", check_frozen())
    keep = ("19095176093", "20374217194", "20342007439", "16028472005")
    df, wins = analyse(HR_MAX_SET, keep)
    df["disp_G_trail"] = track(df, "G_trail", kind="ewma")
    df["disp_G_flat"] = track(df, "G_flat", kind="ewma")
    df["disp_ASCENT_kalman"] = track(df, "ASCENT", "ASCENT_sd", kind="kalman")
    df["disp_ASCENT_race"] = track(df, "ASCENT", "ASCENT_sd", kind="race_aware")
    df.to_csv(os.path.join(RES, "real_estimates_hrmax170.csv"), index=False)
    pd.to_pickle(wins, os.path.join(RES, "real_windows_hrmax170.pkl"))
    races = df[df.race]
    hr_alt = float(np.round(np.percentile(np.concatenate(
        [strava.load(os.path.join(ROOT, "data", "strava", f"{i}.json"),
                     json.load(open(os.path.join(ROOT, "data", "strava_meta.json"))))[0]["hr"]
         for i in races.id]), 99.5)))
    df2, _ = analyse(hr_alt)
    df2.to_csv(os.path.join(RES, f"real_estimates_hrmax{int(hr_alt)}.csv"), index=False)
    txt = summarise(df, "HRmax = 170 (configured)") + "\n" + summarise(df2, f"HRmax = {hr_alt:.0f} (99.5th pct of race HR)")
    open(os.path.join(RES, "real_summary.txt"), "w").write(txt + f"\nhr_alt={hr_alt}\n")
    print(txt)
    # internal consistency: climb-only vs flat-only ASCENT within the same activity
    meta = json.load(open(os.path.join(ROOT, "data", "strava_meta.json")))
    cons = []
    for aid in df["id"]:
        p, info = strava.load(os.path.join(ROOT, "data", "strava", f"{aid}.json"), meta)
        row = dict(id=aid, date=info["date"], surface=info["surface"], race=info["race"])
        for nm, gw in (("climb", (0.04, 1.0)), ("flat", (-0.03, 0.03))):
            a = es.ascent(p, HR_REST, HR_MAX_SET, info["surface"], es.AscentConfig(grade_window=gw, min_windows=8))
            row[nm] = es.value(a)
            row[nm + "_n"] = a.get("n", 0)
        a = es.ascent(p, HR_REST, HR_MAX_SET, info["surface"], es.AscentConfig(use_h=True))
        row["h_free"] = a.get("h", np.nan)
        a = es.ascent(p, HR_REST, HR_MAX_SET, info["surface"], es.AscentConfig(use_drift=False))
        row["ascent_nodrift"] = es.value(a)
        cons.append(row)
    pd.DataFrame(cons).to_csv(os.path.join(RES, "real_consistency.csv"), index=False)
    # robustness checks: fixed tau values; race depression without descents / early part only
    rob = []
    for aid in df["id"]:
        p, info = strava.load(os.path.join(ROOT, "data", "strava", f"{aid}.json"), meta)
        row = dict(id=aid, date=info["date"], surface=info["surface"], race=info["race"])
        for tau in (20.0, 55.0, 120.0, 240.0):
            a = es.ascent(p, HR_REST, HR_MAX_SET, info["surface"],
                          es.AscentConfig(tau_grid=(tau,), tau_prior=(55.0, 1e6)))
            row[f"tau{int(tau)}"] = es.value(a)
        a = es.ascent(p, HR_REST, HR_MAX_SET, info["surface"], es.AscentConfig(exclude_descents=0.15))
        row["exdesc"] = es.value(a)
        n = p["t"].size
        k = int(0.45 * n)
        p2 = {kk: (v[:k] if isinstance(v, np.ndarray) and v.size == n else v) for kk, v in p.items()}
        a = es.ascent(p2, HR_REST, HR_MAX_SET, info["surface"], es.AscentConfig(min_windows=10))
        row["first45"] = es.value(a)
        rob.append(row)
    pd.DataFrame(rob).to_csv(os.path.join(RES, "real_robustness.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df[["date", "name", "surface", "race", "dplus_per_km", "G_flat", "G_trail", "G_trail_hrgap",
              "ASCENT", "ASCENT_sd", "ASCENT_tau", "ASCENT_drift", "share_climb", "share_walk"]].round(2).to_string())
