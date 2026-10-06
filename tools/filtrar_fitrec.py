"""Extrae de endomondoHR.json.gz (FitRec, UCSD) las carreras a pie con pulso y altitud,
en un formato compacto para el estudio de invariancia al terreno de ASCENT.

Uso (Python 3.8+, solo librería estándar + numpy):
    python filtrar_fitrec.py endomondoHR.json.gz
Produce, en la misma carpeta:  fitrec_runs_000.npz, fitrec_runs_001.npz, ...  (≈ 150–250 MB cada uno)
y fitrec_runs_index.csv (una fila por carrera).  Tarda unos 20–40 minutos.

No se conservan las coordenadas GPS: solo distancia acumulada, altitud, pulso y tiempo.
"""
import ast
import csv
import gzip
import json
import math
import os
import sys
import time

import numpy as np

SPORTS = {"run", "running", "trail run", "trail running", "orienteering", "mountain run"}
MAX_POINTS_PER_FILE = 25_000_000           # ≈ 50 000 entrenamientos de 500 puntos


def parse(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        try:
            return json.loads(line.replace("'", '"'))
        except Exception:
            return ast.literal_eval(line)


def haversine_cum(lat, lon):
    r = 6371008.8
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return np.concatenate([[0.0], np.cumsum(2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1))))])


class Writer:
    def __init__(self, out_dir):
        self.out_dir, self.k = out_dir, 0
        self.reset()

    def reset(self):
        self.hr, self.alt, self.dist, self.t, self.n = [], [], [], [], 0

    def add(self, hr, alt, dist, t):
        self.hr.append(hr.astype(np.uint8)); self.alt.append(alt.astype(np.float32))
        self.dist.append(dist.astype(np.float32)); self.t.append(t.astype(np.uint32))
        self.n += hr.size
        if self.n >= MAX_POINTS_PER_FILE:
            self.flush()

    def flush(self):
        if not self.hr:
            return
        fp = os.path.join(self.out_dir, f"fitrec_runs_{self.k:03d}.npz")
        np.savez_compressed(fp, hr=np.concatenate(self.hr), alt=np.concatenate(self.alt),
                            dist=np.concatenate(self.dist), t=np.concatenate(self.t))
        print(f"  escrito {fp} ({os.path.getsize(fp) / 1e6:.0f} MB)", flush=True)
        self.k += 1
        self.reset()


def main(path):
    out_dir = os.path.dirname(os.path.abspath(path))
    w = Writer(out_dir)
    idx = open(os.path.join(out_dir, "fitrec_runs_index.csv"), "w", newline="")
    cw = csv.writer(idx)
    cw.writerow(["id", "userId", "gender", "sport", "start_ts", "n", "file", "offset",
                 "duration_s", "dist_m", "dplus_m", "hr_mean", "hr_max"])
    opener = gzip.open if path.endswith(".gz") else open
    n_all = n_run = n_kept = 0
    t0 = time.time()
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_all += 1
            if n_all % 10000 == 0:
                print(f"{n_all} entrenamientos leídos, {n_kept} carreras guardadas, {time.time() - t0:.0f} s", flush=True)
            try:
                d = parse(line)
            except Exception:
                continue
            if not d or str(d.get("sport", "")).lower() not in SPORTS:
                continue
            n_run += 1
            try:
                hr = np.asarray(d["heart_rate"], float)
                alt = np.asarray(d["altitude"], float)
                ts = np.asarray(d["timestamp"], float)
                lat = np.asarray(d["latitude"], float)
                lon = np.asarray(d["longitude"], float)
            except Exception:
                continue
            n = min(hr.size, alt.size, ts.size, lat.size, lon.size)
            if n < 100:
                continue
            hr, alt, ts, lat, lon = hr[:n], alt[:n], ts[:n], lat[:n], lon[:n]
            ok = np.isfinite(hr) & np.isfinite(alt) & np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon)
            if ok.mean() < 0.9:
                continue
            hr, alt, ts, lat, lon = hr[ok], alt[ok], ts[ok], lat[ok], lon[ok]
            order = np.argsort(ts, kind="stable")
            hr, alt, ts, lat, lon = hr[order], alt[order], ts[order], lat[order], lon[order]
            dur = ts[-1] - ts[0]
            if dur < 600 or dur > 8 * 3600:
                continue
            dist = haversine_cum(lat, lon)
            if dist[-1] < 1500 or dist[-1] / dur > 7.0:      # < 1.5 km o > 25 km/h: no es una carrera a pie
                continue
            if np.mean((hr > 30) & (hr < 230)) < 0.9:
                continue
            hr = np.clip(hr, 0, 255)
            dz = np.diff(alt)
            dplus = float(np.clip(dz, 0, None).sum())
            cw.writerow([d.get("id"), d.get("userId"), d.get("gender"), d.get("sport"), int(ts[0]), hr.size,
                         w.k, w.n, int(dur), round(float(dist[-1])), round(dplus),
                         round(float(hr.mean()), 1), int(hr.max())])
            w.add(hr, alt, dist, ts - ts[0])
            n_kept += 1
    w.flush()
    idx.close()
    print(f"hecho: {n_all} entrenamientos, {n_run} carreras a pie, {n_kept} guardadas, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
