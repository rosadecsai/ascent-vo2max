"""Round-trip tests for the activity readers (TCX and GPX need no extra packages)."""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascent import estimators as es, simulate as sm  # noqa: E402
from ascent.io import read_activity  # noqa: E402

START = dt.datetime(2026, 9, 1, 8, 0, 0, tzinfo=dt.timezone.utc)


def _session():
    rng = np.random.default_rng(3)
    r = sm.random_runner(rng, vo2max=50.0, hr_rest=52.0, hr_max=180.0)
    s = sm.random_session(r, "mountain", False, rng)
    keep = np.arange(0, s.t.size, 4)                      # 4-s sampling, as in many exports
    t = np.round(s.t[keep] - s.t[0])
    return t, np.round(s.hr[keep]), np.round(s.dist[keep], 2), np.round(s.alt[keep], 2), np.round(s.cad[keep])


def _iso(sec):
    return (START + dt.timedelta(seconds=float(sec))).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_tcx(path, t, hr, dist, alt, cad):
    rows = [f"<Trackpoint><Time>{_iso(t[i])}</Time><AltitudeMeters>{alt[i]}</AltitudeMeters>"
            f"<DistanceMeters>{dist[i]}</DistanceMeters><HeartRateBpm><Value>{int(hr[i])}</Value></HeartRateBpm>"
            f"<Extensions><ns3:TPX><ns3:RunCadence>{int(cad[i])}</ns3:RunCadence></ns3:TPX></Extensions></Trackpoint>"
            for i in range(t.size)]
    open(path, "w").write(
        '<?xml version="1.0"?><TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" '
        'xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2"><Activities><Activity Sport="Running">'
        "<Lap><Track>" + "".join(rows) + "</Track></Lap></Activity></Activities></TrainingCenterDatabase>")


def _write_gpx(path, t, hr, dist, alt, cad):
    lat = 37.0 + np.degrees(dist / 6371008.8)              # along a meridian: distance is preserved
    rows = [f'<trkpt lat="{lat[i]:.9f}" lon="-3.4"><ele>{alt[i]}</ele><time>{_iso(t[i])}</time><extensions>'
            f"<gpxtpx:TrackPointExtension><gpxtpx:hr>{int(hr[i])}</gpxtpx:hr><gpxtpx:cad>{2 * int(cad[i])}</gpxtpx:cad>"
            "</gpxtpx:TrackPointExtension></extensions></trkpt>" for i in range(t.size)]
    open(path, "w").write(
        '<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1"><trk><type>trail_running</type>'
        "<trkseg>" + "".join(rows) + "</trkseg></trk></gpx>")


def test_tcx_and_gpx_roundtrip(tmp_path):
    t, hr, dist, alt, cad = _session()
    ref = es.ascent(es.prep(t, hr, dist, alt, cad), 52.0, 180.0, "trail")
    for ext, writer in ((".tcx", _write_tcx), (".gpx", _write_gpx)):
        fp = str(tmp_path / ("a" + ext))
        writer(fp, t, hr, dist, alt, cad)
        a = read_activity(fp)
        assert a["start"] == START
        assert np.allclose(a["dist"] - a["dist"][0], dist - dist[0], atol=0.05)   # GPX distance starts at 0
        assert np.allclose(a["cad"], cad)                  # GPX steps/min converted back to strides/min
        res = es.ascent(es.prep(a["t"], a["hr"], a["dist"], a["alt"], a["cad"]), 52.0, 180.0, "trail")
        assert abs(es.value(res) - es.value(ref)) < 0.05, (ext, es.value(res), es.value(ref))
    assert read_activity(str(tmp_path / "a.gpx"))["surface"] == "trail"
