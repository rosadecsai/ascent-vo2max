"""Loader for the compact FitRec (Endomondo) extract written by tools/filtrar_fitrec.py.

The extract keeps, per running workout, heart rate, altitude, cumulative distance (haversine from
the GPS track) and time; no cadence, no HR anchors, no sport sub-type.  Data: Ni, Muhlstein &
McAuley (2019), academic use only, not redistributed with this code.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import estimators as es

MAX_GAP = 35.0      # s; runs with a mean sample spacing above 30 s are not selected
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fitrec")


class FitRec:
    def __init__(self, folder=DEFAULT_DIR):
        self.folder = folder
        self.index = pd.read_csv(os.path.join(folder, "fitrec_runs_index.csv"))
        self.index["start"] = pd.to_datetime(self.index["start_ts"], unit="s")
        self._files = {}

    def _file(self, k):
        if k not in self._files:
            z = np.load(os.path.join(self.folder, f"fitrec_runs_{k:03d}.npz"))
            self._files[k] = {a: z[a] for a in ("hr", "alt", "dist", "t")}
        return self._files[k]

    def streams(self, row):
        f = self._file(int(row["file"]))
        a, n = int(row["offset"]), int(row["n"])
        sl = slice(a, a + n)
        t = f["t"][sl].astype(float)
        dist = f["dist"][sl].astype(float)
        # GPS position jumps: cap the per-sample speed at 7 m/s and rebuild the cumulative distance
        inc = np.clip(np.diff(dist), 0.0, 7.0 * np.maximum(np.diff(t), 1.0))
        dist = np.concatenate([[0.0], np.cumsum(inc)])
        return dict(t=t, hr=f["hr"][sl].astype(float), dist=dist, alt=f["alt"][sl].astype(float))

    def prep(self, row, walk_heuristic=False):
        s = self.streams(row)
        # Endomondo workouts were resampled to 500 points, so the spacing grows with duration
        # (6 s median, 20 s for a 2.5-h run); the gap flag must allow for it
        return es.prep(s["t"], s["hr"], s["dist"], s["alt"], cad=None, walk_heuristic=walk_heuristic,
                       max_gap=MAX_GAP)
