"""ASCENT v2 development on FitRec with a runner split.

Runners are split by the parity of their Endomondo userId: even = development half, odd = test
half.  Candidate configurations are compared on the development half only (flat and hilly runs);
the chosen one is frozen and then run once on the test half.

    python3 fitrec_v2.py dev          # all candidates on the development half
    python3 fitrec_v2.py test NAME    # one frozen candidate on the test half
    python3 fitrec_v2.py summary dev|test
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ascent import estimators as es  # noqa: E402
from ascent.fitrec import FitRec  # noqa: E402

RES = os.path.join(ROOT, "results")
HR_REST = 55.0
W = 30 * 86400
BINS, LABELS = [8, 15, 25, 40], ["8-15", "15-25", "25-40", ">40"]

CANDIDATES = {
    "v1": dict(surface="road", cfg={}),
    "T": dict(surface="road", cfg=dict(tau_grid=(55.0,))),
    "TP": dict(surface="road", cfg=dict(tau_grid=(20.0, 35.0, 50.0, 65.0, 80.0), tau_prior=(55.0, 15.0))),
    "A": dict(surface="auto", cfg={}),
    "TA": dict(surface="auto", cfg=dict(tau_grid=(55.0,))),
    "TAL75": dict(surface="auto", cfg=dict(tau_grid=(55.0,), ecc_lambda=0.75)),
    "TAL100": dict(surface="auto", cfg=dict(tau_grid=(55.0,), ecc_lambda=1.0)),
    "TAD": dict(surface="auto", cfg=dict(tau_grid=(55.0,), drift_prior=(0.05, 0.03))),   # = ASCENT v2
    # noise diagnostics (not candidates for the frozen configuration)
    "TAW": dict(surface="auto", cfg=dict(tau_grid=(55.0,), use_weights=False)),
    "TAnd": dict(surface="auto", cfg=dict(tau_grid=(55.0,), use_drift=False)),
    "TAh": dict(surface="auto", cfg=dict(tau_grid=(55.0,), huber_c=1.2)),
}


def runs_for(split):
    sel = pd.read_pickle(os.path.join(RES, "fitrec_selected.pkl"))
    sel = sel[(sel.userId % 2 == 0) if split == "dev" else (sel.userId % 2 == 1)]
    if split == "dev":                      # candidates are compared on flat and hilly runs only
        sel = sel[sel.cls != "mid"]
    return sel.reset_index(drop=True)


def job(args):
    rows, names = args
    fr = FitRec()
    out = []
    for r in rows:
        p = fr.prep(r)
        o = dict(id=r["id"], userId=r["userId"])
        for nm in names:
            v = CANDIDATES[nm]
            a = es.ascent(p, HR_REST, float(r["hr_max_u"]), v["surface"], es.AscentConfig(**v["cfg"]))
            o[nm] = a.get("vo2max", np.nan)
            o[nm + "_sd"] = a.get("sd", np.nan)
        out.append(o)
    return out


def run(split, names):
    sel = runs_for(split)
    print(split, len(sel), "runs,", sel.userId.nunique(), "runners,", names, flush=True)
    chunks = [(g.to_dict("records"), names) for _, g in sel.groupby(np.arange(len(sel)) % 24)]
    t0 = time.time()
    rows = []
    with Pool(2) as pool:
        for k, out in enumerate(pool.imap_unordered(job, chunks)):
            rows.extend(out)
            print(f"chunk {k + 1}/24, {len(rows)} runs, {time.time() - t0:.0f} s", flush=True)
    res = sel.merge(pd.DataFrame(rows), on=["id", "userId"])
    fp = os.path.join(RES, f"fitrec_v2_{split}.pkl")
    if os.path.exists(fp):            # keep earlier candidates (same run selection)
        old = pd.read_pickle(fp)
        if len(old) == len(res) and set(old.id) == set(res.id):
            old = old.set_index("id")
            for c in res.columns:
                if c in names or c.replace("_sd", "") in names:
                    old[c] = res.set_index("id")[c].reindex(old.index)
            res = old.reset_index()
    res.to_pickle(fp)
    print("done", f"{time.time() - t0:.0f} s")
    return res


def summary(split):
    df = pd.read_pickle(os.path.join(RES, f"fitrec_v2_{split}.pkl"))
    names = [n for n in CANDIDATES if n in df.columns]
    rows = []
    for nm in names:
        gaps = {b: [] for b in LABELS}
        per_user = []
        for uid, g in df.groupby("userId"):
            flat = g[g.cls == "flat"]
            d = {b: [] for b in LABELS}
            for _, r in g[g.c > 8].iterrows():
                near = flat[np.abs(flat.start_ts - r.start_ts) <= W].dropna(subset=[nm])
                if len(near) == 0 or not np.isfinite(r[nm]):
                    continue
                d[LABELS[min(np.searchsorted(BINS, r.c, side="right") - 1, 3)]].append(r[nm] - near[nm].mean())
            for b in LABELS:
                if d[b]:
                    gaps[b].append(np.mean(d[b]))
        o = dict(candidate=nm, split=split)
        for b in LABELS:
            x = np.asarray(gaps[b])
            o[f"gap{b}"] = stats.trim_mean(x, 0.1) if x.size > 5 else np.nan
            o[f"med{b}"] = np.median(x) if x.size else np.nan
            h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size) if x.size > 2 else np.nan
            o[f"lo{b}"], o[f"hi{b}"] = x.mean() - h, x.mean() + h
            o[f"n{b}"] = x.size
        o["sess_sd_flat"] = df[df.cls == "flat"].groupby("userId")[nm].std().median()
        o["cov_hilly"] = df[df.cls == "hilly"][nm].notna().mean() * 100
        o["cov_flat"] = df[df.cls == "flat"][nm].notna().mean() * 100
        o["mean_flat"] = df[df.cls == "flat"][nm].mean()
        o["stated_sd"] = df[nm + "_sd"].median()
        rows.append(o)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, f"fitrec_v2_{split}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string())
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        run("dev", sys.argv[2:] or list(CANDIDATES))
        summary("dev")
    elif mode == "test":
        run("test", sys.argv[2:])
        summary("test")
    else:
        summary(sys.argv[2])
