"""Analyses added in the first revision (Sports Engineering): uncertainty of the trimmed means,
counts, matching, comparability, coverage, heart-rate anchors, decomposition of v2, sampling.

Reads results/fitrec_desc.pkl, fitrec_estimates.pkl and fitrec_revision.pkl; writes
results/revision/*.csv and results/revision/summary.json (used by make_tables_rev.py).

Uncertainty.  The unit of analysis is the runner.  Every statistic is a 10 % trimmed mean, over
runners, of per-runner mean differences.  Its 95 % interval is a two-level percentile bootstrap:
runners are resampled with replacement and, within each resampled runner, the flat runs and the
non-flat runs are resampled with replacement before the differences are recomputed.  Resampling
whole runners keeps every within-runner dependence (repeated sessions, the same flat runs used as
comparators of several runs); resampling the sessions adds the uncertainty of each runner's own
mean, including that of the comparators.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "revision")
os.makedirs(OUT, exist_ok=True)
W = 30 * 86400
BINS, LABELS = [8, 15, 25, 40], ["8-15", "15-25", "25-40", ">40"]
NB = 2000
TRIM = 0.1


def bin_of(c):
    return LABELS[min(np.searchsorted(BINS, c, side="right") - 1, 3)]


def tmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return stats.trim_mean(x, TRIM) if x.size else np.nan


# ------------------------------------------------------------------------------------------
# per-runner structures
# ------------------------------------------------------------------------------------------
def build(df, e, comp_filter=None, ref=None, valid=None):
    """Per runner: values of the non-flat runs, their bins and the membership matrix of their
    flat comparators (flat runs of the same runner within +-30 days with a valid estimate).

    e: column of the estimator; ref: column used for the flat comparators (default e);
    comp_filter(run_row, flat_frame) -> boolean mask restricting the comparators;
    valid: boolean Series restricting the runs that may be used (targets and comparators)."""
    ref = ref or e
    out = {}
    use = df if valid is None else df[valid]
    for uid, g in use.groupby("userId"):
        flat = g[(g.cls == "flat") & g[ref].notna()]
        tar = g[(g.c > 8) & g[e].notna()]
        if len(flat) == 0 or len(tar) == 0:
            continue
        M = (np.abs(tar.start_ts.values[:, None] - flat.start_ts.values[None, :]) <= W)
        if comp_filter is not None:
            M &= comp_filter(tar, flat)
        keep = M.any(1)
        if not keep.any():
            continue
        tar = tar[keep]
        M = M[keep]
        dt = np.abs(tar.start_ts.values[:, None] - flat.start_ts.values[None, :]) / 86400.0
        out[uid] = dict(y=tar[e].values, b=np.array([bin_of(c) for c in tar.c.values]), M=M.astype(float),
                        f=flat[ref].values, ncomp=M.sum(1), dt=(dt * M).sum(1) / M.sum(1),
                        ids=tar.id.values, n_tar_all=int(((g.c > 8) & g[e].notna()).sum()))
    return out


def runner_means(S, wr=None, wf=None):
    """Per-runner mean difference by bin (optionally with bootstrap weights)."""
    rows = {}
    for uid, s in S.items():
        w_f = np.ones(s["f"].size) if wf is None else wf[uid]
        w_r = np.ones(s["y"].size) if wr is None else wr[uid]
        den = s["M"] @ w_f
        ok = den > 0
        gap = np.full(s["y"].size, np.nan)
        gap[ok] = s["y"][ok] - (s["M"][ok] @ (w_f * s["f"])) / den[ok]
        o = {}
        for b in LABELS:
            m = (s["b"] == b) & ok & (w_r > 0)
            if m.any():
                o[b] = np.average(gap[m], weights=w_r[m])
        rows[uid] = o
    return rows


def stat_by_bin(rows, uids=None):
    uids = list(rows) if uids is None else uids
    out = {}
    for b in LABELS:
        x = np.array([rows[u][b] for u in uids if b in rows[u]])
        out[b] = (tmean(x), x.size)
    return out


def boot(S, seed=0, nb=NB, paired_with=None):
    """Two-level bootstrap of the trimmed mean by bin.  With paired_with (another structure on the
    same runners and runs) the statistic is the trimmed mean of the per-runner paired difference."""
    rng = np.random.default_rng(seed)
    uids = np.array(sorted(S))
    reps = {b: [] for b in LABELS}
    for _ in range(nb):
        samp = rng.choice(uids, uids.size, replace=True)
        wr, wf = {}, {}
        for u in set(samp.tolist()):
            wr[u] = rng.multinomial(S[u]["y"].size, np.full(S[u]["y"].size, 1.0 / S[u]["y"].size)).astype(float)
            wf[u] = rng.multinomial(S[u]["f"].size, np.full(S[u]["f"].size, 1.0 / S[u]["f"].size)).astype(float)
        rows = runner_means({u: S[u] for u in wr}, wr, wf)
        if paired_with is not None:
            rows2 = runner_means({u: paired_with[u] for u in wr if u in paired_with}, wr, wf)
        for b in LABELS:
            if paired_with is None:
                x = [rows[u][b] for u in samp if b in rows[u]]
            else:
                x = [rows[u][b] - rows2[u][b] for u in samp if u in rows2 and b in rows[u] and b in rows2[u]]
            if len(x) >= 5:
                reps[b].append(tmean(x))
    return {b: np.asarray(v) for b, v in reps.items()}


def summarise(S, label, seed=0, nb=NB, extra=None):
    rows = runner_means(S)
    reps = boot(S, seed, nb)
    out = []
    for b in LABELS:
        x = np.array([rows[u][b] for u in rows if b in rows[u]])
        n_runs = int(sum(((s["b"] == b)).sum() for s in S.values()))
        if x.size < 5:
            continue
        lo, hi = np.percentile(reps[b], [2.5, 97.5])
        lo90, hi90 = np.percentile(reps[b], [5, 95])
        h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
        o = dict(label=label, bin=b, trim=tmean(x), lo=lo, hi=hi, lo90=lo90, hi90=hi90, median=np.median(x),
                 mean=x.mean(), mean_lo=x.mean() - h, mean_hi=x.mean() + h, n_runners=int(x.size), n_runs=n_runs,
                 sd_runners=x.std(ddof=1))
        if extra:
            o.update(extra)
        out.append(o)
    return out


def same_runs(Sa, Sb):
    """Restrict two structures built on the same frame to the runs present in both."""
    A, B = {}, {}
    for u in set(Sa) & set(Sb):
        common = np.intersect1d(Sa[u]["ids"], Sb[u]["ids"])
        if common.size == 0:
            continue
        for src, dst in ((Sa, A), (Sb, B)):
            s = src[u]
            k = np.isin(s["ids"], common)
            order = np.argsort(s["ids"][k])
            dst[u] = dict(y=s["y"][k][order], b=s["b"][k][order], M=s["M"][k][order], f=s["f"], ncomp=s["ncomp"][k][order],
                          dt=s["dt"][k][order], ids=s["ids"][k][order])
    return A, B


def paired(df, e1, e2, label, seed=0, nb=NB):
    """Trimmed mean of the per-runner difference between two estimators on the same runs, with the
    same flat comparators (runs and comparators valid for both)."""
    v = df[e1].notna() & df[e2].notna()
    S1, S2 = build(df, e1, valid=v), build(df, e2, valid=v)
    r1, r2 = runner_means(S1), runner_means(S2)
    reps = boot(S1, seed, nb, paired_with=S2)
    out = []
    for b in LABELS:
        x = np.array([r1[u][b] - r2[u][b] for u in r1 if u in r2 and b in r1[u] and b in r2[u]])
        if x.size < 5:
            continue
        lo, hi = np.percentile(reps[b], [2.5, 97.5])
        out.append(dict(label=label, bin=b, trim=tmean(x), lo=lo, hi=hi, n_runners=int(x.size)))
    return out


def windows(df, nb=500):
    """Width of the comparator window: +-15, +-60 and +-90 days instead of +-30 (classic, HR-GAP, v1)."""
    global W
    rows, info = [], {}
    w0 = W
    for days in (15, 30, 60, 90):
        W = days * 86400
        for e in ["G_flat", "G_trail_hrgap", "ASCENT"]:
            S = build(df, e)
            if days != 30:
                rows += summarise(S, f"win{days}|{e}", seed=160, nb=nb)
            if e == "G_flat":
                x = df[(df.c > 8) & df.G_flat.notna()]
                paired_n = int(sum(s_["y"].size for s_ in S.values()))
                dt = np.concatenate([s_["dt"] for s_ in S.values()])
                info[str(days)] = dict(paired=paired_n, pct_paired=float(paired_n / len(x) * 100), dt_med=float(np.median(dt)),
                                       runners_steep=int(sum(1 for s_ in S.values() if (s_["b"] == ">40").any())))
    W = w0
    return rows, info


# ------------------------------------------------------------------------------------------
def main():
    desc = pd.read_pickle(os.path.join(RES, "fitrec_desc.pkl"))
    df = pd.read_pickle(os.path.join(RES, "fitrec_estimates.pkl"))
    rev = pd.read_pickle(os.path.join(RES, "fitrec_revision.pkl"))
    df = df.merge(rev.drop(columns=["userId"]), on="id", how="left")
    df["spacing"] = df.dur / df.n
    df["int"] = (df.hr_med - 55.0) / (df.hr_max_u - 55.0)
    df["bin"] = np.where(df.c <= 8, "flat", [bin_of(c) for c in df.c])
    test = df[df.userId % 2 == 1]
    dev = df[df.userId % 2 == 0]
    J = {}
    tabs = []

    # ---- A. selection flow and participants ------------------------------------------------
    d = desc.copy()
    d["c"] = d.dplus_s / d.dist_s * 1000
    asym = (d.dplus_s - d.dminus_s).abs() / np.maximum(d.dplus_s + d.dminus_s, 1.0)
    steps = [("runs with HR, GPS and altitude", pd.Series(True, index=d.index)),
             ("duration >= 25 min", d.dur >= 1500),
             ("moving > 80 % of the time", d.frac_moving > 0.8),
             ("median HR > 100 bpm", d.hr_med > 100),
             ("mean sample spacing <= 30 s", d.dur / d.n <= 30),
             ("mean altitude -100 to 3500 m", d.alt_mean.between(-100, 3500)),
             ("climbing <= 150 m/km", d.c <= 150),
             ("gain/loss imbalance <= 0.5", asym <= 0.5),
             ("altitude roughness <= 5 m", d.alt_rough <= 5)]
    m = pd.Series(True, index=d.index)
    flow = []
    for name, cond in steps:
        m &= cond.fillna(False)
        flow.append(dict(step=name, runs=int(m.sum()), runners=int(d[m].userId.nunique())))
    flow.append(dict(step=">= 5 flat and >= 5 hilly runs per runner", runs=int(d[m & d.userId.isin(df.userId.unique())].shape[0]),
                     runners=int(df.userId.nunique())))
    flow.append(dict(step="most recent 30 flat, 60 hilly, 20 intermediate", runs=int(len(df)), runners=int(df.userId.nunique())))
    pd.DataFrame(flow).to_csv(os.path.join(OUT, "flow.csv"), index=False)
    J["flow"] = flow
    g = df.groupby("userId")
    yrs = pd.to_datetime(df.start_ts, unit="s").dt.year
    J["participants"] = dict(
        n=int(df.userId.nunique()), n_runs=int(len(df)),
        span_q1_days=float(((g.start_ts.max() - g.start_ts.min()) / 86400).quantile(.25)),
        span_q3_days=float(((g.start_ts.max() - g.start_ts.min()) / 86400).quantile(.75)), male=int((g.gender.first() == "male").sum()), female=int((g.gender.first() == "female").sum()),
        unknown=int((g.gender.first() == "unknown").sum()), runs_med=float(g.size().median()),
        runs_q1=float(g.size().quantile(.25)), runs_q3=float(g.size().quantile(.75)),
        span_med_days=float(((g.start_ts.max() - g.start_ts.min()) / 86400).median()),
        year_min=int(yrs.min()), year_max=int(yrs.max()), share_2013_2015=float(yrs.between(2013, 2015).mean() * 100),
        dur_med=float(df.dur.median() / 60), dur_q1=float(df.dur.quantile(.25) / 60), dur_q3=float(df.dur.quantile(.75) / 60),
        dist_med=float(df.dist_s.median() / 1000), v_med=float(df.v_mean.median()), v_q1=float(df.v_mean.quantile(.25)),
        v_q3=float(df.v_mean.quantile(.75)), hrmed_med=float(df.hr_med.median()),
        hrmax_med=float(g.hr_max_u.first().median()), hrmax_q1=float(g.hr_max_u.first().quantile(.25)),
        hrmax_q3=float(g.hr_max_u.first().quantile(.75)), orienteering=int((df.sport == "orienteering").sum()),
        spacing_med=float(df.spacing.median()), spacing_q3=float(df.spacing.quantile(.75)), spacing_max=float(df.spacing.max()),
        flat_level_classic=float(g.apply(lambda x: x[x.cls == "flat"].G_flat.mean()).median()),
        flat_level_classic_q1=float(g.apply(lambda x: x[x.cls == "flat"].G_flat.mean()).quantile(.25)),
        flat_level_classic_q3=float(g.apply(lambda x: x[x.cls == "flat"].G_flat.mean()).quantile(.75)))
    # descriptive comparability by class/bin
    comp = df.groupby("bin").agg(runs=("id", "size"), runners=("userId", "nunique"), dur_med=("dur", lambda x: x.median() / 60),
                                 int_med=("int", "median"), v_med=("v_mean", "median"), spacing_med=("spacing", "median"),
                                 c_med=("c", "median")).loc[["flat"] + LABELS]
    comp.to_csv(os.path.join(OUT, "bins_descriptive.csv"))
    J["bins_descriptive"] = comp.reset_index().to_dict("records")

    # ---- B. main table: all runners --------------------------------------------------------
    EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT", "ASCENT_gh"]
    S_all = {e: build(df, e) for e in EST}
    for k, e in enumerate(EST):
        tabs += summarise(S_all[e], f"all|{e}", seed=10 + k)
    # coverage: available runs per bin (valid estimate) and runs with / without comparator
    cov = []
    for e in EST + ["v2"]:
        for b in LABELS:
            x = df[df.bin == b]
            valid = x[e].notna()
            S = S_all[e] if e in S_all else build(df, e)
            paired_runs = int(sum((s["b"] == b).sum() for s in S.values()))
            cov.append(dict(estimator=e, bin=b, runs=int(len(x)), valid=int(valid.sum()), valid_pct=float(valid.mean() * 100),
                            with_comparator=paired_runs, no_comparator=int(valid.sum()) - paired_runs))
        x = df[df.cls == "flat"]
        cov.append(dict(estimator=e, bin="flat", runs=int(len(x)), valid=int(x[e].notna().sum()),
                        valid_pct=float(x[e].notna().mean() * 100), with_comparator=np.nan, no_comparator=np.nan))
    pd.DataFrame(cov).to_csv(os.path.join(OUT, "coverage.csv"), index=False)
    J["coverage"] = cov
    # matching statistics (classic estimator: the widest reconstruction sample)
    mt = []
    for e in ["G_flat", "ASCENT"]:
        S = S_all[e]
        for b in LABELS:
            nc = np.concatenate([s["ncomp"][s["b"] == b] for s in S.values()])
            dt = np.concatenate([s["dt"][s["b"] == b] for s in S.values()])
            mt.append(dict(estimator=e, bin=b, runs=int(nc.size), ncomp_med=float(np.median(nc)), ncomp_q1=float(np.percentile(nc, 25)),
                           ncomp_q3=float(np.percentile(nc, 75)), one_comp_pct=float((nc == 1).mean() * 100),
                           dt_med=float(np.median(dt)), dt_q1=float(np.percentile(dt, 25)), dt_q3=float(np.percentile(dt, 75))))
    pd.DataFrame(mt).to_csv(os.path.join(OUT, "matching.csv"), index=False)
    J["matching"] = mt

    pd.DataFrame(tabs).to_csv(os.path.join(OUT, "gaps_partial.csv"), index=False)
    # ---- C. common set of sessions valid for all methods -------------------------------------
    allv = df[["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]].notna().all(axis=1)
    J["common_runs"] = dict(n=int(allv.sum()), pct=float(allv.mean() * 100),
                            by_bin={b: float(allv[df.bin == b].mean() * 100) for b in ["flat"] + LABELS})
    for k, e in enumerate(["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]):
        tabs += summarise(build(df, e, valid=allv), f"common|{e}", seed=30 + k)
    # what the excluded sessions look like (no-descent estimate missing, others present)
    x = df[(df.c >= 25)]
    miss = x.G_trail_nodesc.isna() & x.G_trail_hrgap.notna()
    J["nodesc_missing"] = dict(n=int(miss.sum()), c_med=float(x[miss].c.median()), c_med_kept=float(x[~miss & x.G_trail_nodesc.notna()].c.median()),
                               dur_med=float(x[miss].dur.median() / 60), dur_med_kept=float(x[x.G_trail_nodesc.notna()].dur.median() / 60))
    miss = x.G_flat.isna() & x.ASCENT.notna()
    J["recon_missing"] = dict(n=int(miss.sum()), c_med=float(x[miss].c.median()), c_med_kept=float(x[x.G_flat.notna()].c.median()),
                              v_med=float(x[miss].v_mean.median()), v_med_kept=float(x[x.G_flat.notna()].v_mean.median()),
                              int_med=float(x[miss].int.median()), int_med_kept=float(x[x.G_flat.notna()].int.median()))

    # ---- D. comparable sessions: duration and intensity --------------------------------------
    def comp_filter(tar, flat):
        r = tar.dur.values[:, None] / flat.dur.values[None, :]
        di = np.abs(tar["int"].values[:, None] - flat["int"].values[None, :])
        return (r >= 0.75) & (r <= 1 / 0.75) & (di <= 0.05)
    for k, e in enumerate(["G_flat", "G_trail", "G_trail_hrgap", "G_trail_nodesc", "ASCENT"]):
        tabs += summarise(build(df, e, comp_filter=comp_filter), f"matched|{e}", seed=50 + k)
    tabs += summarise(build(test, "v2", comp_filter=comp_filter), "matched_test|v2", seed=59)
    # within-runner regression with intensity as a further covariate
    sl = []
    for e in EST:
        rows = []
        for uid, gg in df.groupby("userId"):
            gg = gg.dropna(subset=[e])
            if len(gg) < 12 or gg.c.std() < 3:
                continue
            base = [np.ones(len(gg)), gg.c / 10.0, (gg.start_ts - gg.start_ts.mean()) / (365 * 86400), (gg.dur - gg.dur.mean()) / 3600.0]
            b1 = np.linalg.lstsq(np.column_stack(base), gg[e].values, rcond=None)[0]
            b2 = np.linalg.lstsq(np.column_stack(base + [(gg["int"] - gg["int"].mean()) * 10]), gg[e].values, rcond=None)[0]
            rows.append((b1[1], b2[1], b2[4]))
        a = np.array(rows)
        o = dict(estimator=e, n=int(len(a)))
        for j, nm in enumerate(["slope", "slope_int", "per_int10"]):
            xx = a[:, j]
            h = stats.t.ppf(0.975, xx.size - 1) * xx.std(ddof=1) / np.sqrt(xx.size)
            o[nm], o[nm + "_lo"], o[nm + "_hi"] = float(xx.mean()), float(xx.mean() - h), float(xx.mean() + h)
        sl.append(o)
    pd.DataFrame(sl).to_csv(os.path.join(OUT, "slopes.csv"), index=False)
    J["slopes"] = sl

    pd.DataFrame(tabs).to_csv(os.path.join(OUT, "gaps_partial.csv"), index=False)
    # ---- E. heart-rate anchors ----------------------------------------------------------------
    for e in ["G_flat", "G_trail_hrgap", "v2"]:
        tabs += summarise(build(df, e), f"anchor_base|{e}", seed=70, nb=500)
        for a in ["rest45", "rest65", "maxm10", "maxp10"]:
            tabs += summarise(build(df, f"{e}_{a}"), f"anchor_{a}|{e}", seed=70, nb=500)
    J["anchor_levels"] = {e: {a: float(df[df.cls == "flat"][c].mean()) for a, c in
                              [("base", e)] + [(a, f"{e}_{a}") for a in ["rest45", "rest65", "maxm10", "maxp10"]]}
                          for e in ["G_flat", "G_trail_hrgap", "v2"]}
    J["anchor_cov"] = {e: {a: float(df[df.c >= 25][c].notna().mean() * 100) for a, c in
                           [("base", e)] + [(a, f"{e}_{a}") for a in ["rest45", "rest65", "maxm10", "maxp10"]]}
                       for e in ["G_flat", "G_trail_hrgap", "v2"]}

    pd.DataFrame(tabs).to_csv(os.path.join(OUT, "gaps_partial.csv"), index=False)
    # ---- F. decomposition of the revision -----------------------------------------------------
    for split, fr in (("test", test), ("dev", dev), ("allr", df)):
        for k, e in enumerate(["ASCENT", "T_road", "D_road", "v2_road", "v2_trail", "v2", "G_trail_hrgap", "G_trail_nodesc", "G_flat"]):
            tabs += summarise(build(fr, e), f"{split}|{e}", seed=90 + k)
    for a, b_ in (("v2", "G_trail_hrgap"), ("v2", "v2_road"), ("v2_road", "ASCENT"), ("v2_road", "G_trail_hrgap"), ("v2", "ASCENT")):
        tabs += paired(test, a, b_, f"paired_test|{a}-{b_}", seed=120)
    J["split"] = dict(test_runners=int(test.userId.nunique()), dev_runners=int(dev.userId.nunique()),
                      test_runs=int(len(test)), dev_runs=int(len(dev)),
                      h_mean_by_bin={b: float(test[test.bin == b].v2_h.mean()) for b in ["flat"] + LABELS})
    # session-to-session SD within runner on flat runs (paired across estimators)
    sd = {}
    for split, fr in (("test", test), ("allr", df)):
        f = fr[fr.cls == "flat"]
        s = f.groupby("userId")[["v2", "G_trail_hrgap", "ASCENT", "G_flat", "v2_road", "T_road", "D_road", "v2_trail", "G_trail_nodesc"]].std()
        s = s.dropna()
        o = dict(n=int(len(s)), **{c: float(s[c].median()) for c in s.columns})
        for a, b_ in (("v2", "G_trail_hrgap"), ("v2", "ASCENT"), ("ASCENT", "G_trail_hrgap")):
            dd = s[a] - s[b_]
            o[f"{a}-{b_}_med"] = float(dd.median())
            o[f"{a}-{b_}_p"] = float(stats.wilcoxon(dd).pvalue)
            o[f"{a}-{b_}_frac"] = float((dd > 0).mean() * 100)
            rng = np.random.default_rng(5)
            bs = [np.median(rng.choice(dd.values, dd.size)) for _ in range(5000)]
            o[f"{a}-{b_}_lo"], o[f"{a}-{b_}_hi"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
        sd[split] = o
    J["session_sd"] = sd
    J["v2_valid_test"] = float(test[test.c >= 25].v2.notna().mean() * 100)
    J["valid_hilly"] = {split: {e: float(fr[fr.c >= 25][e].notna().mean() * 100) for e in
                                ["ASCENT", "T_road", "D_road", "v2_road", "v2_trail", "v2", "G_trail_hrgap", "G_trail_nodesc", "G_flat"]}
                        for split, fr in (("test", test), ("dev", dev), ("allr", df))}
    steep = df[df.c >= 40]
    J["diag"] = dict(steep_runs=int(len(steep)), steep_runners=int(steep.userId.nunique()),
                     steep_runs_test=int((steep.userId % 2 == 1).sum()), steep_runners_test=int(steep[steep.userId % 2 == 1].userId.nunique()))
    # runs with an estimate but without a flat comparator within +-30 days (classic reconstruction)
    x = df[(df.c > 8) & df.G_flat.notna()]
    paired_n = int(sum(s_["y"].size for s_ in S_all["G_flat"].values()))
    J["no_comparator"] = dict(valid=int(len(x)), paired=paired_n, pct=float((len(x) - paired_n) / len(x) * 100))
    nc_all = np.concatenate([s_["ncomp"] for s_ in S_all["G_flat"].values()])
    dt_all = np.concatenate([s_["dt"] for s_ in S_all["G_flat"].values()])
    J["matching_all"] = dict(ncomp_med=float(np.median(nc_all)), ncomp_q1=float(np.percentile(nc_all, 25)), ncomp_q3=float(np.percentile(nc_all, 75)),
                             dt_med=float(np.median(dt_all)), dt_q1=float(np.percentile(dt_all, 25)), dt_q3=float(np.percentile(dt_all, 75)))

    # ---- G. sampling interval ------------------------------------------------------------------
    med = df.spacing.median()
    for nm, fr in (("fine", df[df.spacing <= 7]), ("coarse", df[df.spacing > 7])):
        for e in ["G_flat", "G_trail_hrgap", "ASCENT"]:
            tabs += summarise(build(fr, e), f"spacing_{nm}|{e}", seed=140, nb=500)
    J["spacing"] = dict(median=float(med), fine_runs=int((df.spacing <= 7).sum()), coarse_runs=int((df.spacing > 7).sum()),
                        by_bin={b: float(df[df.bin == b].spacing.median()) for b in ["flat"] + LABELS})

    # ---- H. runner-level-only bootstrap, for comparison ---------------------------------------
    rows = runner_means(S_all["G_flat"])
    rng = np.random.default_rng(1)
    x = np.array([rows[u][">40"] for u in rows if ">40" in rows[u]])
    bs = [tmean(rng.choice(x, x.size)) for _ in range(5000)]
    J["runner_only_boot"] = dict(lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

    # ---- I. width of the comparator window ---------------------------------------------------
    rows, J["windows"] = windows(df)
    tabs += rows

    T = pd.DataFrame(tabs)
    T.to_csv(os.path.join(OUT, "gaps.csv"), index=False)
    json.dump(J, open(os.path.join(OUT, "summary.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(T.round(2).to_string())
    print(json.dumps(J, indent=1, default=str)[:6000])


def add_windows():
    """Add the comparator-window analysis to existing results without recomputing the rest."""
    df = pd.read_pickle(os.path.join(RES, "fitrec_estimates.pkl"))
    rows, info = windows(df)
    T = pd.read_csv(os.path.join(OUT, "gaps.csv"))
    T = pd.concat([T[~T.label.str.startswith("win")], pd.DataFrame(rows)], ignore_index=True)
    T.to_csv(os.path.join(OUT, "gaps.csv"), index=False)
    J = json.load(open(os.path.join(OUT, "summary.json")))
    J["windows"] = info
    json.dump(J, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(pd.DataFrame(rows)[["label", "bin", "trim", "lo", "hi", "n_runners", "n_runs"]].round(2).to_string())
    print(info)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "windows":
        add_windows()
    else:
        main()
