"""Physiological building blocks shared by the simulator and the estimators.

All metabolic quantities are per kg body mass.  Gradient ``i`` is rise/run (tan of the
slope angle), as in Minetti et al. (2002).  Energy costs are *net* (above rest) and are
expressed per metre travelled along the path (belt distance).
"""
from __future__ import annotations

import numpy as np

# --- constants -------------------------------------------------------------------------
E_O2 = 20.9          # J per ml O2 (energy equivalent used by Minetti et al. 2002)
VO2_REST = 3.5       # ml/kg/min (1 MET)
ACSM_NET = 0.2       # ml/kg per metre: ACSM level-running net cost  -> 4.18 J/kg/m
C0_RUN = ACSM_NET * E_O2           # 4.18 J/kg/m: population level-running cost
MINETTI_RUN0 = 3.6                 # Minetti level-running cost (elite mountain runners)
MINETTI_WALK0 = 2.5                # Minetti level-walking cost at optimal speed
SCALE = C0_RUN / MINETTI_RUN0      # rescales Minetti's curves to the ACSM level cost
ALT_SLOPE = 0.063                  # VO2max loss per 1000 m above 300 m (Wehrlin & Hallen 2006)
ALT_REF = 300.0

I_MIN, I_MAX = -0.45, 0.45         # validity range of Minetti's polynomials


def minetti_run(i):
    """Net energy cost of running, J/kg/m (Minetti et al. 2002, eq. for C_r)."""
    i = np.clip(np.asarray(i, float), I_MIN, I_MAX)
    return 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6


def minetti_walk(i):
    """Net energy cost of walking, J/kg/m (Minetti et al. 2002, eq. for C_w)."""
    i = np.clip(np.asarray(i, float), I_MIN, I_MAX)
    return 280.5 * i**5 - 58.7 * i**4 - 76.8 * i**3 + 51.9 * i**2 + 19.6 * i + 2.5


def cost_run(i):
    """Population running cost (J/kg/m) = Minetti shape rescaled to the ACSM level cost."""
    return SCALE * minetti_run(i)


def cost_walk(i):
    return SCALE * minetti_walk(i)


def rel_cost_run(i):
    """Minetti grade factor C_r(i)/C_r(0): the 'metabolic' grade-adjusted-pace factor."""
    return minetti_run(i) / MINETTI_RUN0


def strava_like_factor(i):
    """Heart-rate-equivalent grade factor in the spirit of Strava's 2017 GAP model.

    Uphill it follows Minetti; downhill it has a shallow minimum of 0.88 at -9 % and is back
    to 1.0 at -18 % (Robb 2017), rising slowly beyond.  Used (i) by the simulator to
    represent the heart-rate excess seen on descents and (ii) by a Garmin-like variant.
    """
    i = np.asarray(i, float)
    up = rel_cost_run(np.maximum(i, 0.0))
    par = 1.0 - 0.12 * (1.0 - ((i + 0.09) / 0.09) ** 2)
    steep = 1.0 + 1.0 * (-0.18 - i)
    down = np.where(i >= -0.18, par, steep)
    return np.where(i >= 0, up, down)


def vo2_from_power(p_wkg):
    """Net metabolic power (W/kg) -> gross VO2 (ml/kg/min)."""
    return VO2_REST + 60.0 * np.asarray(p_wkg, float) / E_O2


def acsm_level_vo2(v_ms):
    """ACSM level running: VO2 = 3.5 + 0.2 * v[m/min]."""
    return VO2_REST + ACSM_NET * 60.0 * np.asarray(v_ms, float)


def altitude_factor(z_m, slope=ALT_SLOPE):
    """Fraction of sea-level VO2max available at altitude z (linear model, >= 0.5)."""
    z = np.asarray(z_m, float)
    return np.clip(1.0 - slope * np.maximum(z - ALT_REF, 0.0) / 1000.0, 0.5, 1.0)


def hrr_fraction(hr, hr_rest, hr_max):
    """Heart-rate-reserve fraction (Swain & Leutholtz 1997: %HRR ~ %VO2R)."""
    return (np.asarray(hr, float) - hr_rest) / (hr_max - hr_rest)


def vo2max_from_point(vo2, hr, hr_rest, hr_max):
    """Single-point extrapolation along the %HRR = %VO2R line."""
    f = hrr_fraction(hr, hr_rest, hr_max)
    return VO2_REST + (np.asarray(vo2, float) - VO2_REST) / f
