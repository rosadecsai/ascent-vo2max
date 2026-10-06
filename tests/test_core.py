"""Sanity tests: physiology formulas and recovery of VO2max in idealised worlds."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascent import physiology as ph, simulate as sm, estimators as es  # noqa: E402

IDEAL = dict(gps_bias=False, gps_noise=False, terrain=False, ecc=False, drift=False,
             altitude=False, stops=False, econ_mismatch=False, hr_noise=False, baro_noise=False,
             obstacles=False, arousal=False, curvature=False)


def test_minetti_values():
    assert abs(ph.minetti_run(0.0) - 3.6) < 1e-12
    assert abs(ph.minetti_walk(0.0) - 2.5) < 1e-12
    assert abs(ph.rel_cost_run(0.0) - 1.0) < 1e-12
    # optimum downhill gradient of running cost lies between -0.30 and -0.15
    i = np.linspace(-0.45, 0.0, 901)
    assert -0.30 < i[np.argmin(ph.minetti_run(i))] < -0.15
    # steep-uphill efficiency approaches ~0.2-0.25 (Minetti 2002)
    eff = 9.81 * 0.3 / np.sqrt(1 + 0.3**2) / ph.minetti_run(0.3)
    assert 0.2 < eff < 0.26


def test_strava_like_and_altitude():
    assert abs(ph.strava_like_factor(-0.09) - 0.88) < 1e-9
    assert abs(ph.strava_like_factor(-0.18) - 1.0) < 1e-9
    assert abs(ph.strava_like_factor(0.1) - ph.rel_cost_run(0.1)) < 1e-12
    assert abs(ph.altitude_factor(300) - 1.0) < 1e-12
    assert abs(ph.altitude_factor(1300) - (1 - 0.063)) < 1e-12


def test_acsm_consistency():
    v = 3.0
    net_acsm = ph.acsm_level_vo2(v) - ph.VO2_REST
    net_cost = 60 * ph.cost_run(0.0) * v / ph.E_O2
    assert abs(net_acsm - net_cost) < 1e-9


def _ideal_session(kind, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng, econ=1.0, up_eff=1.0, down_mult=1.0, walk_econ=1.0, gamma=1.0)
    c = sm.make_course(kind, rng)
    o = dict(IDEAL); o.update(extra or {})
    s = sm.simulate_session(r, c, 0.75, False, rng, opts=o)
    return r, es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)


def test_flat_road_recovery():
    for seed in range(3):
        r, p = _ideal_session("road_flat", seed)
        g = es.garmin_like(p, r.hr_rest, r.hr_max, "flat")
        cfg = es.AscentConfig(use_altitude=False)
        a = es.ascent(p, r.hr_rest, r.hr_max, "road", cfg)
        assert abs(g["vo2max"] - r.target) < 1.0, (g["vo2max"], r.target)
        assert abs(a["vo2max"] - r.target) < 1.0, (a["vo2max"], r.target)


def test_mountain_recovery_ascent_ideal():
    cfg = es.AscentConfig(use_altitude=False, h_prior={"road": (1.0, 0.03), "trail": (1.0, 0.06)},
                          use_descent_model=False)
    for seed in range(3):
        r, p = _ideal_session("mountain", seed)
        a = es.ascent(p, r.hr_rest, r.hr_max, "trail", cfg)
        assert abs(a["vo2max"] - r.target) < 1.5, (a["vo2max"], r.target)


def test_kalman():
    k = es.KalmanTracker(q_per_day=0.0, r_floor=0.0)
    for d, z in enumerate([50, 52, 48, 50]):
        k.update(d, z, 2.0)
    assert abs(k.m - 50.0) < 1e-9


def test_v2_ideal_and_auto_surface():
    """v2 recovers VO2max in the idealised mountain world and 'auto' gives the trail prior there."""
    over = dict(use_altitude=False, h_prior={"road": (1.0, 0.03), "trail": (1.0, 0.06)}, use_descent_model=False)
    for seed in range(3):
        r, p = _ideal_session("mountain", seed)
        a = es.ascent_v2(p, r.hr_rest, r.hr_max, "trail", **over)
        assert a["tau"] == 55.0
        assert abs(a["vo2max"] - r.target) < 1.5, (a["vo2max"], r.target)
    r, p = _ideal_session("mountain", 0)
    assert es.climb_per_km(p) > 40
    assert abs(es.ascent_v2(p, r.hr_rest, r.hr_max, "auto")["h_prior"] - 1.07) < 1e-9
    r, p = _ideal_session("road_flat", 0)
    assert abs(es.ascent_v2(p, r.hr_rest, r.hr_max, "auto")["h_prior"] - 1.0) < 1e-9
