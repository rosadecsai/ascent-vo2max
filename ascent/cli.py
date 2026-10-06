"""Run ASCENT and the Garmin-like reconstructions on exported activities.

    python3 -m ascent.cli FILE [FILE ...] --hr-rest 52 --hr-max 170
                          [--surface auto|road|trail] [--race] [--history my_vo2max.csv]

FILE: .fit (Garmin "Export Original", needs `pip install fitdecode`), .tcx, .gpx or Strava-stream
.json.  --surface auto takes the surface from the file when it says (FIT sub-sport "trail",
GPX type "trail_running") and otherwise lets the terrain prior follow the climbing of the run (v2).
ASCENT v2 (default; --v1 for the original): fixed 55-s HR lag, climbing-based terrain prior when
the surface is unknown, tighter drift prior, no gait labels without cadence.  With --history, every analysed activity
is added to (or updated in) a CSV file, and the race-aware Kalman tracker is run over the
whole history; edit the `race` column of that file to mark races afterwards.

The ASCENT configuration is the one frozen after the simulation study; nothing is fitted to
the user.  Values are estimates with the limitations described in the report.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys

import numpy as np

from . import estimators as es
from .io import read_activity

TRAIL_ETA_HAT = 1.035       # Garmin-like accelerometer terrain factor assumed on trails
FIELDS = ["start", "date", "file", "surface", "race", "km", "dplus_m", "minutes", "ascent", "ascent_sd",
          "tau_s", "drift_pct_h", "share_climb", "share_flat", "share_descent", "share_walk",
          "garmin_like_classic", "garmin_like_trail", "garmin_like_nodescent", "garmin_like_hrgap",
          "hr_p995"]


def analyse(path, hr_rest, hr_max, surface="auto", race=False, version=2):
    a = read_activity(path)
    # the labelled surface wins; without a label, v2 interpolates the terrain prior by climbing
    surf = surface if surface != "auto" else (a["surface"] or ("auto" if version == 2 else "road"))
    p = es.prep(a["t"], a["hr"], a["dist"], a["alt"], a["cad"])
    p["eta_hat"][:] = TRAIL_ETA_HAT if surf == "trail" else 1.0
    res = es.ascent_v2(p, hr_rest, hr_max, surf) if version == 2 else es.ascent(p, hr_rest, hr_max, surf)
    ws = res.get("weight_share", {})
    out = dict(start=a["start"].isoformat(timespec="seconds"), date=a["start"].date().isoformat(),
               file=os.path.basename(path), surface=surf, race=int(bool(race)),
               km=a["dist"][-1] / 1000.0, dplus_m=float(np.clip(np.diff(p["zs"]), 0, None).sum()),
               minutes=a["t"][-1] / 60.0, ascent=res.get("vo2max", np.nan), ascent_sd=res.get("sd", np.nan),
               tau_s=res.get("tau", np.nan), drift_pct_h=100 * res.get("drift", np.nan),
               share_climb=ws.get("climb", np.nan), share_flat=ws.get("flat", np.nan),
               share_descent=ws.get("descent", np.nan), share_walk=ws.get("walk", np.nan),
               hr_p995=float(np.percentile(a["hr"], 99.5)))
    for v, k in (("flat", "classic"), ("trail", "trail"), ("trail_nodesc", "nodescent"),
                 ("trail_hrgap", "hrgap")):
        out["garmin_like_" + k] = es.garmin_like(p, hr_rest, hr_max, v).get("vo2max", np.nan)
    out["_surface_assumed"] = surface == "auto" and a["surface"] is None
    out["_h_prior"] = res.get("h_prior", np.nan)
    out["_if_valid"] = res.get("vo2max_if_valid", np.nan)
    return out


def _fmt(x, nd=1):
    return "  --" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def report(o, hr_max):
    src = (f" (not in the file: terrain prior {o['_h_prior']:.3f} from the climbing; use --surface road|trail if known)"
           if o["_surface_assumed"] and np.isfinite(o["_h_prior"]) else
           (" (assumed: the file does not say; use --surface)" if o["_surface_assumed"] else ""))
    print(f"\n{o['date']}  {o['file']}  {o['surface']}{src}{'  RACE' if o['race'] else ''}")
    print(f"  {o['km']:.1f} km, D+ {o['dplus_m']:.0f} m, {o['minutes']:.0f} min")
    if np.isfinite(o["ascent"]):
        print(f"  ASCENT       {o['ascent']:.1f} +/- {o['ascent_sd']:.1f}  (HR lag {o['tau_s']:.0f} s, "
              f"drift {o['drift_pct_h']:+.1f} %/h; weight from climbs {100 * o['share_climb']:.0f} %, "
              f"flats {100 * o['share_flat']:.0f} %, descents {100 * o['share_descent']:.0f} %, "
              f"hiking {100 * o['share_walk']:.0f} %)")
    else:
        extra = f" (unreliable fit: {o['_if_valid']:.1f})" if np.isfinite(o["_if_valid"]) else ""
        print(f"  ASCENT       no valid estimate{extra}")
    print(f"  Garmin-like  classic {_fmt(o['garmin_like_classic'])} | trail {_fmt(o['garmin_like_trail'])} | "
          f"no-descent {_fmt(o['garmin_like_nodescent'])} | HR-GAP {_fmt(o['garmin_like_hrgap'])}")
    if o["hr_p995"] > hr_max:
        print(f"  note: recorded HR reaches {o['hr_p995']:.0f} bpm (99.5th percentile), above the HRmax "
              f"setting of {hr_max:.0f}; an HRmax set too low lowers every HR-based estimate.")


def update_history(path, rows):
    old = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                old[(r["start"], r["file"])] = r
    for o in rows:
        key = (o["start"], o["file"])
        prev = old.get(key)
        rec = {k: o[k] for k in FIELDS}
        if prev is not None and str(prev.get("race", "0")).strip() in ("1", "true", "True", "yes"):
            rec["race"] = 1                 # keep a race mark made by hand
        old[key] = rec
    recs = sorted(old.values(), key=lambda r: r["start"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    return recs


def tracked(recs):
    trk = es.RaceAwareTracker()
    d0 = dt.date.fromisoformat(recs[0]["date"])
    shown, n = None, 0
    for r in recs:
        z, sd = float(r["ascent"] or "nan"), float(r["ascent_sd"] or "nan")
        if not np.isfinite(z):
            continue
        race = str(r["race"]).strip() in ("1", "true", "True", "yes")
        shown = trk.update((dt.date.fromisoformat(r["date"]) - d0).days, z, sd, race=race)
        n += 1
    return shown, n, trk


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--hr-rest", type=float, required=True)
    ap.add_argument("--hr-max", type=float, required=True)
    ap.add_argument("--surface", choices=["auto", "road", "trail"], default="auto")
    ap.add_argument("--race", action="store_true", help="mark these activities as races")
    ap.add_argument("--history", help="CSV file that accumulates sessions for the tracked value")
    ap.add_argument("--v1", action="store_true", help="use the original (v1) configuration instead of v2")
    a = ap.parse_args(argv)
    rows = []
    for fp in a.files:
        try:
            o = analyse(fp, a.hr_rest, a.hr_max, a.surface, a.race, version=1 if a.v1 else 2)
        except Exception as e:  # keep going with the other files
            print(f"\n{os.path.basename(fp)}: skipped ({e})", file=sys.stderr)
            continue
        report(o, a.hr_max)
        rows.append(o)
    if a.history and rows:
        recs = update_history(a.history, rows)
        shown, n, trk = tracked(recs)
        if shown is not None:
            b = trk.b
            print(f"\nTracked VO2max (race-aware Kalman over {n} sessions in {a.history}): {shown:.1f}"
                  + (f"   learned race-day offset {b:+.1f}" if b is not None and abs(b) > 0.05 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
