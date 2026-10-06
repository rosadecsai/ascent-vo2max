"""Terrain-invariance analysis of the FitRec estimates (within-runner comparisons).

For each estimator:
  (a) paired gap: every hilly run minus the mean of the same runner's flat runs within +-30 days;
      averaged per runner, then across runners (mean, 95 % CI, sign test);
  (b) within-runner regression of the estimate on climbing per km, with a linear time trend and
      duration as covariates; the slope per 10 m/km averaged across runners (>= 12 valid runs);
  (c) coverage (share of runs with an estimate) by terrain class.
Writes results/fitrec_effects.csv and the numbers used by the report.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT", "ASCENT_gh"]
WINDOW = 30 * 86400


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan, x.size, np.nan
    m = x.mean()
    h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    p = stats.wilcoxon(x).pvalue if x.size >= 6 else np.nan
    return m, m - h, m + h, x.size, p


def paired_gaps(df, e, lo_cls="flat", hi_cls="hilly"):
    out = []
    for uid, g in df.groupby("userId"):
        g = g.dropna(subset=[e])
        flat = g[g.cls == lo_cls]
        hil = g[g.cls == hi_cls]
        diffs = []
        for _, r in hil.iterrows():
            near = flat[np.abs(flat.start_ts - r.start_ts) <= WINDOW]
            if len(near) >= 1:
                diffs.append(r[e] - near[e].mean())
        if len(diffs) >= 2:
            out.append(dict(userId=uid, gap=np.mean(diffs), n_pairs=len(diffs)))
    return pd.DataFrame(out)


def within_slopes(df, e, min_runs=12):
    out = []
    for uid, g in df.groupby("userId"):
        g = g.dropna(subset=[e])
        if len(g) < min_runs or g.c.std() < 3:
            continue
        X = np.column_stack([np.ones(len(g)), g.c / 10.0, (g.start_ts - g.start_ts.mean()) / (365 * 86400),
                             (g.dur - g.dur.mean()) / 3600.0])
        beta, *_ = np.linalg.lstsq(X, g[e].values, rcond=None)
        out.append(dict(userId=uid, slope10=beta[1], per_hour=beta[3], n=len(g)))
    return pd.DataFrame(out)


def main(fp=os.path.join(RES, "fitrec_estimates.pkl")):
    df = pd.read_pickle(fp)
    df = df[df.get("err").isna()] if "err" in df else df
    rows = []
    for e in EST:
        pg = paired_gaps(df, e)
        m, lo, hi, n, p = ci(pg.gap)
        sl = within_slopes(df, e)
        ms, los, his, ns, ps = ci(sl.slope10)
        md, lod, hid, _, _ = ci(sl.per_hour)
        cov = df.groupby("cls")[e].apply(lambda x: x.notna().mean() * 100)
        rows.append(dict(estimator=e, gap=m, gap_lo=lo, gap_hi=hi, n_users=n, gap_p=p, gap_median=pg.gap.median(),
                         frac_neg=(pg.gap < 0).mean() * 100,
                         slope10=ms, slope_lo=los, slope_hi=his, n_users_slope=ns, slope_p=ps,
                         per_hour=md, per_hour_lo=lod, per_hour_hi=hid,
                         cov_flat=cov.get("flat", np.nan), cov_mid=cov.get("mid", np.nan), cov_hilly=cov.get("hilly", np.nan),
                         mean_flat=df[df.cls == "flat"][e].mean(), mean_hilly=df[df.cls == "hilly"][e].mean()))
        pg.to_pickle(os.path.join(RES, f"fitrec_gaps_{e}.pkl"))
        sl.to_pickle(os.path.join(RES, f"fitrec_slopes_{e}.pkl"))
    fx = pd.DataFrame(rows)
    fx.to_csv(os.path.join(RES, "fitrec_effects.csv"), index=False)
    pd.set_option("display.width", 250)
    print(fx.round(2).to_string())
    # steeper subset: hilly >= 40 m/km
    df2 = df.copy()
    df2["cls"] = np.where(df2.c <= 8, "flat", np.where(df2.c >= 40, "hilly", "mid"))
    rows2 = []
    for e in EST:
        pg = paired_gaps(df2, e)
        m, lo, hi, n, p = ci(pg.gap)
        rows2.append(dict(estimator=e, gap40=m, gap40_lo=lo, gap40_hi=hi, n_users40=n, p40=p))
    fx2 = pd.DataFrame(rows2)
    fx2.to_csv(os.path.join(RES, "fitrec_effects_steep.csv"), index=False)
    print(fx2.round(2).to_string())
    dose(df)
    noise_floor(df)
    return fx


BINS, LABELS = [8, 15, 25, 40], ["8-15", "15-25", "25-40", ">40"]


def dose(df):
    """Per run above 8 m/km: estimate minus the runner's flat runs within +-30 days, by climb bin;
    per-runner means, then the median and 10 % trimmed mean across runners."""
    rows = []
    for uid, g in df.groupby("userId"):
        flat = g[g.cls == "flat"]
        for _, r in g[g.c > 8].iterrows():
            near = flat[np.abs(flat.start_ts - r.start_ts) <= WINDOW]
            if len(near) == 0:
                continue
            d = dict(userId=uid, c=r.c, bin=LABELS[min(np.searchsorted(BINS, r.c, side="right") - 1, 3)])
            for e in EST:
                d[e] = r[e] - near[e].mean()
            rows.append(d)
    dr = pd.DataFrame(rows)
    dr.to_pickle(os.path.join(RES, "fitrec_dose.pkl"))
    out = []
    for b in LABELS:
        x = dr[dr.bin == b]
        for e in EST:
            u = x.groupby("userId")[e].mean().dropna()
            m, lo, hi, n, p = ci(u)
            out.append(dict(bin=b, estimator=e, n_users=n, median=u.median(), trim=stats.trim_mean(u, 0.1),
                            mean=m, lo=lo, hi=hi, p=p, n_runs=x[e].notna().sum()))
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(RES, "fitrec_dose_summary.csv"), index=False)
    for v in ("median", "trim"):
        print(v)
        print(t.pivot(index="bin", columns="estimator", values=v).loc[LABELS][EST].round(2))


def noise_floor(df, seed=0, n_splits=50):
    """Flat-versus-flat control: each runner's flat runs split at random in two halves; the SD
    of the per-runner gap is averaged over n_splits random splits."""
    rng = np.random.default_rng(seed)
    sds = []
    flats = {uid: g[g.cls == "flat"] for uid, g in df.groupby("userId")}
    for _ in range(n_splits):
        rows = []
        for uid, flat in flats.items():
            if len(flat) < 4:
                continue
            idx = rng.permutation(len(flat))
            a, b = flat.iloc[idx[: len(idx) // 2]], flat.iloc[idx[len(idx) // 2:]]
            rows.append({e: a[e].mean() - b[e].mean() for e in EST})
        sds.append(pd.DataFrame(rows).std())
    c_sd = pd.concat(sds, axis=1).mean(axis=1)
    sd_sess = df[df.cls == "flat"].groupby("userId")[EST].std().median()
    out = pd.DataFrame(dict(control_sd=c_sd, session_sd=sd_sess))
    out.to_csv(os.path.join(RES, "fitrec_noise.csv"))
    print(out.round(2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "fitrec_estimates.pkl"))
