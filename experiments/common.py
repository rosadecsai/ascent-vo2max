"""Shared helpers for the experiments."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascent import simulate as sm, estimators as es  # noqa: E402

KINDS = ["road_flat", "road_hilly", "trail_runnable", "mountain", "skyrun"]
SURF = {"road_flat": "road", "road_hilly": "road", "trail_runnable": "trail",
        "mountain": "trail", "skyrun": "trail"}
LABEL = {"road_flat": "Flat road", "road_hilly": "Hilly road", "trail_runnable": "Runnable trail",
         "mountain": "Mountain trail", "skyrun": "Sky/alpine"}


def run_estimators(s, runner, kind, cfg=None, variants=("flat", "trail", "trail_hrgap", "trail_nodesc"),
                   hr_rest=None, hr_max=None):
    p = es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)
    hr_rest = runner.hr_rest if hr_rest is None else hr_rest
    hr_max = runner.hr_max if hr_max is None else hr_max
    out = {}
    for v in variants:
        g = es.garmin_like(p, hr_rest, hr_max, v)
        out["G_" + v] = g.get("vo2max", np.nan)
        out["G_" + v + "_n"] = g.get("n", 0)
    a = es.ascent(p, hr_rest, hr_max, SURF[kind], cfg)
    out["ASCENT"] = a.get("vo2max", np.nan)
    out["ASCENT_sd"] = a.get("sd", np.nan)
    out["ASCENT_tau"] = a.get("tau", np.nan)
    out["ASCENT_drift"] = a.get("drift", np.nan)
    out["ASCENT_n"] = a.get("n", 0)
    ws = a.get("weight_share", {})
    for k in ("climb", "flat", "descent", "walk"):
        out["share_" + k] = ws.get(k, np.nan)
    return out
