"""Read single activities exported from a watch or from Strava.

Supported formats
  .fit   Garmin original file (Garmin Connect: "Export Original"; the .zip it downloads can be
         passed directly); needs `pip install fitdecode`
  .tcx   Training Center XML (Garmin Connect: "Export to TCX")
  .gpx   GPX with Garmin TrackPointExtension (hr, cad); distance is rebuilt from the coordinates
  .json  Strava streams as saved by this project (keys time, heart_rate, distance, altitude, cadence)

Every reader returns a dict with
  t (s from start), hr (bpm), dist (m), alt (m), cad (strides/min, one foot; or None),
  start (datetime, UTC), sport (str, may be ""), surface ("trail" / "road" / None if the file
  does not say).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _parse_time(s):
    s = s.strip().replace("Z", "+00:00")
    d = _dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


def _haversine(lat, lon):
    r = 6371008.8
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return np.concatenate([[0.0], np.cumsum(2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1))))])


def _fill(y):
    """Linear interpolation over missing values (NaN); edges take the nearest value."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.all() or not ok.any():
        return y
    i = np.arange(y.size)
    return np.interp(i, i[ok], y[ok])


def _finish(ts, hr, dist, alt, cad, sport="", surface=None):
    ts = np.asarray(ts, float)
    hr, dist, alt = (np.asarray(x, float) for x in (hr, dist, alt))
    cad = None if cad is None else np.asarray(cad, float)
    order = np.argsort(ts, kind="stable")
    ts, hr, dist, alt = ts[order], hr[order], dist[order], alt[order]
    cad = None if cad is None else cad[order]
    # keep strictly increasing timestamps
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, hr, dist, alt = ts[keep], hr[keep], dist[keep], alt[keep]
    cad = None if cad is None else cad[keep]
    if not np.isfinite(alt).any():
        raise ValueError("the file has no altitude data")
    if not np.isfinite(dist).any():
        raise ValueError("the file has no distance or position data")
    hr[hr <= 30] = np.nan                                   # optical-HR dropouts
    if np.isfinite(hr).mean() < 0.5:
        raise ValueError("the file has heart rate in fewer than half of its samples")
    hr, alt = _fill(hr), _fill(alt)
    dist = np.maximum.accumulate(_fill(dist))
    if cad is not None:
        if not np.isfinite(cad).any():
            cad = None
        else:
            cad = _fill(cad)
            mv = np.gradient(dist) > 0.5
            if mv.any() and np.nanmedian(cad[mv]) > 120:    # steps/min -> strides/min (one foot)
                cad = cad / 2.0
    start = _EPOCH + _dt.timedelta(seconds=float(ts[0]))
    return dict(t=ts - ts[0], hr=hr, dist=dist, alt=alt, cad=cad, start=start,
                sport=sport or "", surface=surface)


# ------------------------------------------------------------------------------------------
def read_fit(path):
    try:
        import fitdecode
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading .fit files needs the fitdecode package: pip install fitdecode") from e
    ts, hr, dist, alt, cad, lat, lon = [], [], [], [], [], [], []
    sport, sub = "", ""
    semi = 180.0 / 2 ** 31
    with fitdecode.FitReader(path) as fr:
        for fm in fr:
            if fm.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            if fm.name in ("session", "sport") and not sport:
                sport = str(fm.get_value("sport", fallback="") or "")
                sub = str(fm.get_value("sub_sport", fallback="") or "")
            if fm.name != "record":
                continue
            t = fm.get_value("timestamp", fallback=None)
            if t is None:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=_dt.timezone.utc)
            ts.append((t - _EPOCH).total_seconds())
            g = lambda *names: next((v for v in (fm.get_value(n, fallback=None) for n in names)
                                     if v is not None), None)
            hr.append(g("heart_rate"))
            dist.append(g("distance"))
            alt.append(g("enhanced_altitude", "altitude"))
            c = g("cadence")
            fc = g("fractional_cadence")
            cad.append(None if c is None else c + (fc or 0.0))
            la, lo = g("position_lat"), g("position_long")
            lat.append(None if la is None else la * semi)
            lon.append(None if lo is None else lo * semi)
    f = lambda x: np.array([np.nan if v is None else float(v) for v in x])
    dist_a = f(dist)
    if not np.isfinite(dist_a).any() and np.isfinite(f(lat)).sum() > 1:
        ok = np.isfinite(f(lat)) & np.isfinite(f(lon))
        dist_a = np.full(len(ts), np.nan)
        dist_a[ok] = _haversine(f(lat)[ok], f(lon)[ok])
    surface = "trail" if "trail" in sub else ("road" if sport == "running" else None)
    return _finish(ts, f(hr), dist_a, f(alt), f(cad), f"{sport}/{sub}".strip("/"), surface)


def read_tcx(path):
    root = ET.parse(path).getroot()
    sport = ""
    for el in root.iter():
        if _local(el.tag) == "Activity":
            sport = el.attrib.get("Sport", "")
            break
    ts, hr, dist, alt, cad, lat, lon = [], [], [], [], [], [], []
    for tp in root.iter():
        if _local(tp.tag) != "Trackpoint":
            continue
        rec = dict(Time=None, AltitudeMeters=np.nan, DistanceMeters=np.nan, hr=np.nan, cad=np.nan,
                   lat=np.nan, lon=np.nan)
        for el in tp.iter():
            name = _local(el.tag)
            if name == "HeartRateBpm":
                for v in el:
                    if _local(v.tag) == "Value" and (v.text or "").strip():
                        rec["hr"] = float(v.text)
                continue
            txt = (el.text or "").strip()
            if not txt:
                continue
            if name == "Time":
                rec["Time"] = _parse_time(txt)
            elif name in ("AltitudeMeters", "DistanceMeters"):
                rec[name] = float(txt)
            elif name in ("RunCadence", "Cadence"):
                rec["cad"] = float(txt)
            elif name == "LatitudeDegrees":
                rec["lat"] = float(txt)
            elif name == "LongitudeDegrees":
                rec["lon"] = float(txt)
        if rec["Time"] is None:
            continue
        ts.append((rec["Time"] - _EPOCH).total_seconds())
        hr.append(rec["hr"]); dist.append(rec["DistanceMeters"]); alt.append(rec["AltitudeMeters"])
        cad.append(rec["cad"]); lat.append(rec["lat"]); lon.append(rec["lon"])
    dist = np.asarray(dist, float)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if not np.isfinite(dist).any() and np.isfinite(lat).sum() > 1:
        ok = np.isfinite(lat) & np.isfinite(lon)
        dist = np.full(lat.size, np.nan)
        dist[ok] = _haversine(lat[ok], lon[ok])
    return _finish(ts, hr, dist, alt, cad, sport, None)


def read_gpx(path):
    root = ET.parse(path).getroot()
    kind = ""
    for el in root.iter():
        if _local(el.tag) == "type" and el.text:
            kind = el.text.strip()
            break
    ts, hr, alt, cad, lat, lon = [], [], [], [], [], []
    for pt in root.iter():
        if _local(pt.tag) != "trkpt":
            continue
        rec = dict(time=None, ele=np.nan, hr=np.nan, cad=np.nan)
        for el in pt.iter():
            name, txt = _local(el.tag), (el.text or "").strip()
            if not txt:
                continue
            if name == "time":
                rec["time"] = _parse_time(txt)
            elif name == "ele":
                rec["ele"] = float(txt)
            elif name in ("hr", "heartrate"):
                rec["hr"] = float(txt)
            elif name in ("cad", "cadence"):
                rec["cad"] = float(txt)
        if rec["time"] is None:
            continue
        ts.append((rec["time"] - _EPOCH).total_seconds())
        lat.append(float(pt.attrib["lat"])); lon.append(float(pt.attrib["lon"]))
        hr.append(rec["hr"]); alt.append(rec["ele"]); cad.append(rec["cad"])
    order = np.argsort(ts, kind="stable")
    lat, lon = np.asarray(lat)[order], np.asarray(lon)[order]
    ts = np.asarray(ts)[order]
    hr, alt, cad = (np.asarray(x, float)[order] for x in (hr, alt, cad))
    dist = _haversine(lat, lon)
    surface = "trail" if "trail" in kind.lower() else None
    return _finish(ts, hr, dist, alt, cad, kind, surface)


def read_strava_json(path):
    d = json.load(open(path))
    t = np.asarray(d["time"], float)
    start = d.get("start_date")
    base = (_parse_time(start) - _EPOCH).total_seconds() if start else 0.0
    cad = d.get("cadence")
    return _finish(t + base, d["heart_rate"], d["distance"], d["altitude"], cad, d.get("sport_type", ""), None)


def read_activity(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":                     # Garmin Connect "Export Original" gives a .zip with the .fit
        import tempfile
        import zipfile
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if os.path.splitext(n)[1].lower() in (".fit", ".tcx", ".gpx")]
            if not names:
                raise ValueError("the .zip file contains no .fit, .tcx or .gpx file")
            with tempfile.TemporaryDirectory() as tmp:
                return read_activity(z.extract(names[0], tmp))
    if ext == ".fit":
        return read_fit(path)
    if ext == ".tcx":
        return read_tcx(path)
    if ext == ".gpx":
        return read_gpx(path)
    if ext == ".json":
        return read_strava_json(path)
    raise ValueError(f"unsupported file type: {ext} (use .fit, .zip, .tcx, .gpx or .json)")
