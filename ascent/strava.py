"""Load Strava activity streams (as saved from the Strava API) into estimator inputs."""
from __future__ import annotations

import json
import os

import numpy as np

from . import estimators as es

RACES = {  # activities that were races, from their names/descriptions
    "16402119854": "Carrera Universidad-Ciudad 10K 2025",
    "16028472005": "Benemérita Trail 12K 2025",
    "17885746460": "Grazalema Trail Nature (short) 2026",
    "19095176093": "CxM RinRan Mountain 2026",
    "19659497712": "CxM nocturna Albondón 2026",
    "20342007439": "Trail nocturno Dehesa de Santa Fe 2026",
}


def load(path_streams, meta):
    aid = os.path.basename(path_streams)[:-5]
    d = json.load(open(path_streams))
    a = meta[aid]
    t = np.asarray(d["time"], float)
    hr = np.asarray(d["heart_rate"], float)
    dist = np.asarray(d["distance"], float)
    alt = np.asarray(d["altitude"], float)
    cad = np.asarray(d["cadence"], float) if "cadence" in d else None
    # optical-HR dropouts occasionally appear as zeros
    good = hr > 30
    t, hr, dist, alt = t[good], hr[good], dist[good], alt[good]
    cad = cad[good] if cad is not None else None
    p = es.prep(t, hr, dist, alt, cad)
    surface = "road" if a["sport_type"] in ("Run", "Workout") else "trail"
    s = a["summary"]
    info = dict(id=aid, date=a["start_local"][:10], name=a["name"], sport=a["sport_type"],
                surface=surface, race=aid in RACES, race_name=RACES.get(aid, ""),
                dist_km=s["distance"] / 1000.0, dplus=s["elevation_gain"],
                moving_min=s["moving_time"] / 60.0)
    info["dplus_per_km"] = info["dplus"] / max(info["dist_km"], 0.1)
    mv = p["moving"]
    info["frac_descent"] = float(np.mean(p["grade"][mv] < -0.05)) if mv.any() else np.nan
    info["frac_climb"] = float(np.mean(p["grade"][mv] > 0.05)) if mv.any() else np.nan
    info["frac_walk"] = float(np.mean(p["walk"][mv])) if mv.any() else np.nan
    info["alt_mean"] = float(np.mean(p["zs"]))
    info["hr_mean"] = float(np.mean(p["hr"][mv])) if mv.any() else np.nan
    info["hr_p99"] = float(np.percentile(p["hr"], 99))
    return p, info
