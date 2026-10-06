"""Write LaTeX tables and a numbers file (macros) straight from the result files,
so that every number quoted in the report comes from the code."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABEL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "report", "tables")
os.makedirs(TAB, exist_ok=True)
ORDER = ["road_flat", "road_hilly", "trail_runnable", "mountain", "skyrun"]
EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
EST_HDR = ["Classic", "Trail", "No-descent", "HR-GAP", "ASCENT"]
MIN_N = 10          # cells with fewer valid estimates are reported as n/a
NUM = {}


def f(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "--"
    if round(float(x), nd) == 0:
        x = 0.0                      # no "-0.0"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def num(name, val, nd=1, sign=False):
    if isinstance(val, (float, np.floating)):
        NUM[name] = f(val, nd, sign).replace("$-$", r"\ensuremath{-}").replace("+", r"\ensuremath{+}")
    else:
        NUM[name] = str(val)


# ----------------------------------------------------------------------------------------
def tab_main():
    df = pd.read_pickle(os.path.join(RES, "main.pkl"))
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{Garmin-like reconstructions} & \\ \cmidrule(lr){3-6}",
             r"Course & Session & " + " & ".join(EST_HDR) + r" \\",
             r"\midrule"]
    for k in ORDER:
        for race in (False, True):
            d = df[(df.kind == k) & (df.race == race)]
            cells = []
            for e in EST:
                err = (d[e] - d["target"]).dropna()
                if len(err) < MIN_N:
                    cells.append("n/a")
                    continue
                rmse = np.sqrt(np.mean(err**2))
                cells.append(f"{f(err.mean())} ({err.std():.1f}) [{rmse:.1f}]")
            lines.append(f"{LABEL[k] if not race else ''} & {'race' if race else 'training'} & " + " & ".join(cells) + r" \\")
        if k != ORDER[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_main.tex"), "w").write("\n".join(lines))
    # numbers for the text
    for k in ORDER:
        for e in EST:
            err = (df[df.kind == k][e] - df[df.kind == k]["target"]).dropna()
            ok = len(err) >= MIN_N
            num(f"bias{e.replace('_', '')}{k.replace('_', '')}", float(err.mean()) if ok else np.nan, 1, True)
            num(f"rmse{e.replace('_', '')}{k.replace('_', '')}", float(np.sqrt(np.mean(err**2))) if ok else np.nan, 1)
    z = (df["ASCENT"] - df["target"]) / df["ASCENT_sd"]
    num("covOne", float(np.nanmean(np.abs(z) < 1) * 100), 0)
    num("covTwo", float(np.nanmean(np.abs(z) < 2) * 100), 0)
    num("nRunnersMain", str(df.seed.nunique()))
    num("nSessionsMain", str(len(df)))
    sh = df.groupby("kind")[["share_climb", "share_walk", "share_descent", "ASCENT_tau", "ASCENT_drift"]].mean()
    num("shareClimbMountain", float(sh.loc["mountain", "share_climb"] * 100), 0)
    num("shareClimbSky", float(sh.loc["skyrun", "share_climb"] * 100), 0)
    num("shareDescMountain", float(sh.loc["mountain", "share_descent"] * 100), 0)
    num("shareWalkSky", float(sh.loc["skyrun", "share_walk"] * 100), 0)
    num("shareWalkMountain", float(sh.loc["mountain", "share_walk"] * 100), 0)
    num("missGtrail", float(df["G_trail"].isna().mean() * 100), 1)
    for k in ORDER:
        num(f"missGnodesc{k.replace('_', '')}", float(df[df.kind == k]["G_trail_nodesc"].isna().mean() * 100), 0)
    # tau at the 120-s cap in simulation
    for k in ("mountain", "skyrun", "road_flat"):
        num(f"simTauCap{k.replace('_', '')}", float((df[df.kind == k]["ASCENT_tau"] >= 120).mean() * 100), 0)
    num("missAscent", float(df["ASCENT"].isna().mean() * 100), 1)


def tab_ablation():
    out = []
    for nm in ("ablation", "ablation_matched"):
        fp = os.path.join(RES, nm + ".pkl")
        if not os.path.exists(fp):
            continue
        d = pd.read_pickle(fp)
        d["err"] = d["est"] - d["target"]
        g = d.groupby(["variant", "kind"])["err"]
        t = pd.DataFrame({"bias": g.mean(), "rmse": g.apply(lambda x: np.sqrt(np.nanmean(x**2)))})
        out.append((nm, t))
    kinds = ["road_flat", "trail_runnable", "mountain", "skyrun"]
    variants = list(pd.read_pickle(os.path.join(RES, "ablation.pkl"))["variant"].drop_duplicates())
    lines = [r"\begin{tabular}{l" + "rr" * len(kinds) + "}", r"\toprule",
             " & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{{LABEL[k]}}}" for k in kinds]) + r" \\",
             " ".join([rf"\cmidrule(lr){{{2 + 2 * j}-{3 + 2 * j}}}" for j in range(len(kinds))]),
             "Variant & " + " & ".join(["bias & RMSE"] * len(kinds)) + r" \\", r"\midrule"]
    for nm, t in out:
        title = ("Full simulated world" if nm == "ablation" else
                 "World without unmodelled terrain/GPS cost (terrain and GPS shortening switched off, $h$ prior centred at 1)")
        lines.append(rf"\multicolumn{{{1 + 2 * len(kinds)}}}{{l}}{{\emph{{{title}}}}} \\")
        for v in variants:
            cells = []
            for k in kinds:
                b, r = t.loc[(v, k), "bias"], t.loc[(v, k), "rmse"]
                cells += [f(b), f"{r:.2f}"]
            lines.append(v + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_ablation.tex"), "w").write("\n".join(lines))
    t = dict(out)["ablation"]
    for v, key in (("ASCENT (full)", "Full"), ("no HR kinetics", "NoLag"), ("no gait model", "NoGait"),
                   ("no error-budget weights", "NoW"), ("no altitude correction", "NoAlt"),
                   ("no drift term", "NoDrift"), ("no descent model", "NoDesc"),
                   ("free h (self-calibrated)", "FreeH")):
        for k in kinds:
            num(f"abl{key}{k.replace('_', '')}", float(t.loc[(v, k), "rmse"]), 2)
            num(f"ablB{key}{k.replace('_', '')}", float(t.loc[(v, k), "bias"]), 2, True)
    if "ablation_matched" in dict(out):
        t = dict(out)["ablation_matched"]
        for v, key in (("ASCENT (full)", "Full"), ("no HR kinetics", "NoLag"), ("no gait model", "NoGait"),
                       ("free h (self-calibrated)", "FreeH")):
            for k in kinds:
                num(f"ablM{key}{k.replace('_', '')}", float(t.loc[(v, k), "rmse"]), 2)
                num(f"ablMB{key}{k.replace('_', '')}", float(t.loc[(v, k), "bias"]), 2, True)


def tab_sens():
    df = pd.read_pickle(os.path.join(RES, "sensitivity.pkl"))
    d = df[df.kind == "mountain"]
    for e in ("G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"):
        d = d.assign(**{"e_" + e: d[e] - d["target"]})
    for fac in d.factor.unique():
        for li, lv in enumerate(sorted(d[d.factor == fac].level.unique())):
            x = d[(d.factor == fac) & (d.level == lv)]
            tag = fac.replace("_", "") + "LMH"[li]
            for e in ("G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"):
                num(f"sens{e.replace('_', '')}{tag}", float(x["e_" + e].mean()), 1, True)


KEYMAP = {"G_flat": "Gflat", "G_trail@1.035": "Gtrail", "G_trail@1.07": "GtrailH", "G_trail_nodesc@1.035": "Gnodesc",
          "G_trail_hrgap@1.035": "Ghrgap", "ASCENT@1.07": "ASCENT", "ASCENT@1.035": "ASCENTmid",
          "ASCENT@1.0": "ASCENTlow"}
WORLDTAG = {"baseline": "Base", "uphill cost +10 %": "TiltUp", "uphill cost -10 %": "TiltDown",
            "descent HR excess linear in grade": "EccLin", "steeper altitude loss (quadratic)": "AltQuad",
            "accelerating drift (quadratic)": "DriftQuad", "overlapping walk/run cadence": "CadOverlap",
            "all of the above (+10 %)": "AllMis"}


def tab_real():
    if not os.path.exists(os.path.join(RES, "real_estimates_hrmax170.csv")):
        return                          # case-study data are not public
    a = pd.read_csv(os.path.join(RES, "real_estimates_hrmax170.csv"), dtype={"id": str})
    hr_alt = [fn for fn in os.listdir(RES) if fn.startswith("real_estimates_hrmax") and not fn.endswith("170.csv")][0]
    halt = hr_alt.replace("real_estimates_hrmax", "").replace(".csv", "")
    num("hrAlt", halt)
    fx = pd.read_csv(os.path.join(RES, "real_effects.csv"))
    rows_def = [("G_flat", "Garmin-like classic"), ("G_trail@1.035", "Garmin-like trail"),
                ("G_trail_nodesc@1.035", "Garmin-like no-descent"), ("G_trail_hrgap@1.035", "Garmin-like HR-GAP"),
                ("ASCENT@1.07", "ASCENT (prior $h=1.07$)"), ("ASCENT@1.035", "ASCENT with $h=1.035$"),
                ("ASCENT@1.0", "ASCENT with $h=1.00$")]
    cols = [("all", False, "All activities"), ("recent", False, "Jun--Oct 2026"),
            ("recent", True, "Jun--Oct 2026, no alpine day")]

    def cell(r, kind="gap"):
        m, lo, hi = r[kind], r[kind + "_lo"] if kind == "gap" else r["race_lo"], r[kind + "_hi"] if kind == "gap" else r["race_hi"]
        return f"{f(m)} [{f(lo)}, {f(hi)}]"
    lines = [r"\begin{tabular}{l" + "r" * (len(cols) + 1) + "}", r"\toprule",
             r" & \multicolumn{" + str(len(cols)) + r"}{c}{Trail minus road training} & Trail races minus road \\",
             r"\cmidrule(lr){2-" + str(len(cols) + 1) + r"}\cmidrule(lr){" + str(len(cols) + 2) + "-" + str(len(cols) + 2) + "}",
             "Estimator & " + " & ".join(c[2] for c in cols) + r" & Jun--Oct 2026 \\", r"\midrule"]
    for hm in (170, int(halt)):
        lines.append(rf"\multicolumn{{{len(cols) + 2}}}{{l}}{{\emph{{HR$_{{\max}}$ = {hm} bpm}}}} \\")
        for key, lab in rows_def:
            cells = []
            for per, excl, _ in cols:
                r = fx[(fx.hrmax == hm) & (fx.period == per) & (fx.excl_alpine == excl) & (fx.estimator == key)].iloc[0]
                cells.append(cell(r))
                tag = f"{'All' if per == 'all' else 'Recent'}{'Ex' if excl else ''}{'Set' if hm == 170 else 'Alt'}"
                kk = KEYMAP[key]
                num(f"gap{kk}{tag}", float(r["gap"]), 1, True)
                num(f"gapLo{kk}{tag}", float(r["gap_lo"]), 1, True)
                num(f"gapHi{kk}{tag}", float(r["gap_hi"]), 1, True)
                num(f"gapP{kk}{tag}", float(r["gap_p"]), 2)
            r = fx[(fx.hrmax == hm) & (fx.period == "recent") & (fx.excl_alpine == False) & (fx.estimator == key)].iloc[0]
            cells.append(cell(r, "race_gap"))
            num(f"raceGap{kk}{'Set' if hm == 170 else 'Alt'}", float(r["race_gap"]), 1, True)
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
        # macros only (not in the table)
        for key in ("G_trail@1.07",):
            for per, excl, _ in cols:
                r = fx[(fx.hrmax == hm) & (fx.period == per) & (fx.excl_alpine == excl) & (fx.estimator == key)].iloc[0]
                tag = f"{'All' if per == 'all' else 'Recent'}{'Ex' if excl else ''}{'Set' if hm == 170 else 'Alt'}"
                num(f"gap{KEYMAP[key]}{tag}", float(r["gap"]), 1, True)
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_real.tex"), "w").write("\n".join(lines))
    # groups sizes
    num("nRealRecentRoad", str(int(((a.surface == "road") & ~a.race & (a.date >= "2026-06-01")).sum())))
    num("nRealRecentTrail", str(int(((a.surface == "trail") & ~a.race & (a.date >= "2026-06-01")).sum())))
    num("nRealRecentRace", str(int(((a.surface == "trail") & a.race & (a.date >= "2026-06-01")).sum())))
    # per-race differences from the recent road-training mean (ASCENT and Garmin-like trail)
    rec = a[a.date >= "2026-06-01"]
    road = rec[(rec.surface == "road") & ~rec.race]
    for rid, tag in (("19659497712", "Albondon"), ("20342007439", "Dehesa"), ("19095176093", "RinRan")):
        x = a[a.id == rid].iloc[0]
        for e, et in (("ASCENT", "ASCENT"), ("G_trail", "Gtrail")):
            num(f"race{tag}{et}", float(x[e] - road[e].mean()), 1, True)
    # alpine day
    x = a[a.id == "19750627336"].iloc[0]
    num("alpineASCENT", float(x["ASCENT"] - road["ASCENT"].mean()), 1, True)
    num("alpineGtrail", float(x["G_trail"] - road["G_trail"].mean()), 1, True)
    # recent trail SD vs road SD for ASCENT
    tr = rec[(rec.surface == "trail") & ~rec.race]
    num("sdTrailASCENT", float(tr["ASCENT"].std()), 1)
    num("sdRoadASCENT", float(road["ASCENT"].std()), 1)
    # RinRan without drift
    c = pd.read_csv(os.path.join(RES, "real_consistency.csv"), dtype={"id": str})
    rr = c[c.id == "19095176093"].iloc[0]
    num("rinranNoDrift", float(rr["ascent_nodrift"]), 1)
    num("rinranASCENT", float(a[a.id == "19095176093"].iloc[0]["ASCENT"]), 1)
    num("rinranGtrail", float(a[a.id == "19095176093"].iloc[0]["G_trail"]), 1)
    num("rinranDrift", float(a[a.id == "19095176093"].iloc[0]["ASCENT_drift"]), 2)
    num("roadRaceWindows", str(int(a[a.id == "16402119854"].iloc[0]["G_trail_n"])))
    # per-activity appendix table
    rows = [r"\begin{longtable}{lp{4.4cm}lrrrrrr}", r"\toprule",
            r"Date & Activity & Type & D+/km & Classic & Trail & No-desc. & ASCENT & $\sigma$ \\", r"\midrule", r"\endhead"]
    for _, r in a.iterrows():
        typ = ("race" if r["race"] else "") + ("" if not r["race"] else ", ") + r["surface"]
        nm = (r["race_name"] if r["race"] else r["name"]).replace("&", r"\&").replace("#", "")[:42]
        nm = nm.replace("@", "")
        fmt = lambda v: f"{v:.1f}" if np.isfinite(v) else "--"
        rows.append(f"{r['date']} & {nm} & {typ} & {r['dplus_per_km']:.0f} & {fmt(r['G_flat'])} & {fmt(r['G_trail'])} & "
                    f"{fmt(r['G_trail_nodesc'])} & " + (f"{r['ASCENT']:.1f} & {r['ASCENT_sd']:.1f}" if np.isfinite(r['ASCENT']) else "-- & --") + r" \\")
    rows += [r"\bottomrule", r"\end{longtable}"]
    open(os.path.join(TAB, "tab_real_activities.tex"), "w").write("\n".join(rows))
    num("nReal", str(len(a)))
    num("nRealRoad", str(int(((a.surface == "road") & ~a.race).sum())))
    num("nRealTrail", str(int(((a.surface == "trail") & ~a.race).sum())))
    num("nRealRace", str(int(a.race.sum())))
    num("dplusTrailMin", float(a[(a.surface == "trail") & ~a.race]["dplus_per_km"].min()), 0)
    num("dplusTrailMax", float(a[(a.surface == "trail") & ~a.race]["dplus_per_km"].max()), 0)


def sens_macros():
    df = pd.read_pickle(os.path.join(RES, "sensitivity.pkl"))
    d = df[df.kind == "mountain"].copy()
    ests = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
    for e in ests:
        d["e_" + e] = d[e] - d["target"]
    g = d.groupby(["factor", "level"])[["e_" + e for e in ests]].mean()
    for e in ests:
        et = e.replace("_", "")
        num(f"shiftHrmaxLow{et}", float(g.loc[("hrmax_err", -10.0), "e_" + e] - g.loc[("hrmax_err", 0.0), "e_" + e]), 1, True)
        num(f"shiftHrmaxHigh{et}", float(g.loc[("hrmax_err", 10.0), "e_" + e] - g.loc[("hrmax_err", 0.0), "e_" + e]), 1, True)
        for fac in d.factor.unique():
            x = g.loc[fac, "e_" + e]
            num(f"range{et}{fac.replace('_', '')}", float(x.max() - x.min()), 1)
        keep = [fc for fc in d.factor.unique() if fc not in ("terrain_scale", "hrmax_err")]
        vals = np.concatenate([g.loc[fc, "e_" + e].values for fc in keep])
        num(f"sensMin{et}", float(vals.min()), 1, True)
        num(f"sensMax{et}", float(vals.max()), 1, True)


def tab_mismatch():
    fp = os.path.join(RES, "mismatch.pkl")
    if not os.path.exists(fp):
        return
    df = pd.read_pickle(fp)
    ests = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
    worlds = list(df["world"].drop_duplicates())
    kinds = ["road_flat", "trail_runnable", "mountain", "skyrun"]
    # classic, trail and HR-GAP share one window selection, hence one coverage
    for k in kinds:
        same = df[df.kind == k][["G_flat", "G_trail", "G_trail_hrgap"]].notna()
        assert (same.nunique(axis=1) == 1).all(), "Garmin-like coverages differ"
    pretty = {w: w.replace("+10 %", r"$+10$\,\%").replace("-10 %", r"$-10$\,\%")
              .replace("(+10 %)", r"($+10$\,\%)") for w in worlds}
    ncol = 2 + len(ests)
    lines = [r"\begin{tabular}{@{}lr" + "r" * len(ests) + "@{}}", r"\toprule",
             r"World / course & Valid (\%) & " + " & ".join(EST_HDR) + r" \\", r"\midrule"]
    for i, w in enumerate(worlds):
        wt = WORLDTAG[w]
        if i:
            lines.append(r"\addlinespace[3pt]")
        lines.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{pretty[w]}}}}} \\")
        for k in kinds:
            d = df[(df.world == w) & (df.kind == k)]
            kt = k.replace("_", "")
            cov_g = d["G_trail"].notna().mean() * 100
            cov_nd = d["G_trail_nodesc"].notna().mean() * 100
            num(f"mmCov{wt}{kt}", float(cov_g), 0)
            num(f"mmCovND{wt}{kt}", float(cov_nd), 0)
            cells = [f"{cov_g:.0f}/{cov_nd:.0f}"]
            for e in ests:
                err = (d[e] - d["target"]).dropna()
                et = e.replace("_", "")
                ok = len(err) >= MIN_N
                cells.append(f"{f(err.mean())} [{np.sqrt(np.mean(err**2)):.1f}]" if ok else "n/a")
                num(f"mm{wt}{kt}{et}", float(err.mean()) if ok else np.nan, 1, True)
                num(f"mmR{wt}{kt}{et}", float(np.sqrt(np.mean(err**2))) if ok else np.nan, 1)
            # paired comparison on the sessions where both HR-GAP and ASCENT gave an estimate
            c = d.dropna(subset=["G_trail_hrgap", "ASCENT"])
            for e in ("G_trail_hrgap", "ASCENT"):
                ec = c[e] - c["target"]
                num(f"mmC{wt}{kt}{e.replace('_', '')}", float(ec.mean()), 1, True)
                num(f"mmCR{wt}{kt}{e.replace('_', '')}", float(np.sqrt(np.mean(ec**2))), 1)
            # ASCENT on the sessions the Garmin-like methods dropped
            dr = d[d["G_trail_hrgap"].isna()]
            num(f"mmRDrop{wt}{kt}", float(np.sqrt(np.mean((dr["ASCENT"] - dr["target"]) ** 2)))
                if len(dr) >= MIN_N else np.nan, 1)
            row = rf"\quad {LABEL[k]} & " + " & ".join(cells) + r" \\"
            lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    for ln in lines:                                       # an unescaped % would comment out the rest of a row
        assert "%" not in ln.replace(r"\%", ""), ln
    open(os.path.join(TAB, "tab_mismatch.tex"), "w").write("\n".join(lines))
    # worst-case ASCENT bias across worlds and courses
    vals = []
    for w in worlds:
        for k in kinds:
            d = df[(df.world == w) & (df.kind == k)]
            vals.append((d["ASCENT"] - d["target"]).mean())
    num("mmAscentMin", float(min(vals)), 1, True)
    num("mmAscentMax", float(max(vals)), 1, True)
    num("nMismatch", str(df.seed.nunique()))


def tab_long():
    fp = os.path.join(RES, "longitudinal_tracks.pkl")
    if not os.path.exists(fp):
        return
    o = pd.read_pickle(fp)
    tk = {"Garmin-like trail + constant gain": "Garmin", "Garmin-like no-descent + constant gain": "GarminND",
          "ASCENT + Kalman": "Kalman", "ASCENT + race-aware Kalman": "Race"}
    for sc, stag in (("race_effect", ""), ("no_race_effect", "NoRE")):
        d = o[o.scenario == sc]
        for k, kt in tk.items():
            x = d[d.tracker == k]
            num(f"longRmse{kt}{stag}", float(np.sqrt(np.nanmean(x[x.week >= 2]["err"] ** 2))), 1)
            num(f"longBias{kt}{stag}", float(np.nanmean(x[x.week >= 2]["err"])), 1, True)
            for c, ct in (("mountain trail", "Mtn"), ("race", "Race"), ("road", "Road"), ("runnable trail", "Run")):
                y = x[x.cat == c]["change"]
                num(f"longChg{ct}{kt}{stag}", float(y.mean()), 2, True)
                num(f"longDrop{ct}{kt}{stag}", float((y <= -1).mean() * 100), 0)
    num("nLongRunners", str(o.seed.nunique() // 1 if "scenario" not in o else o[o.scenario == "race_effect"].seed.nunique()))


def tab_attr():
    fp = os.path.join(RES, "attribution.pkl")
    if not os.path.exists(fp):
        return
    df = pd.read_pickle(fp)
    for e in ("G_trail", "ASCENT", "G_flat", "G_trail_nodesc"):
        df["e_" + e] = df[e] - df["target"]
    base = df[df.cond == "all"].set_index(["seed", "kind"])
    for kind in df.kind.unique():
        for cond in df.cond.unique():
            if not cond.startswith("no_"):
                continue
            d = df[(df.cond == cond) & (df.kind == kind)].set_index(["seed", "kind"])
            for e in ("G_trail", "ASCENT", "G_flat", "G_trail_nodesc"):
                v = float(np.nanmean(base.loc[d.index, "e_" + e].values - d["e_" + e].values))
                num(f"attr{e.replace('_', '')}{kind.replace('_', '')}{cond[3:].replace('_', '')}", v, 1, True)
        for e in ("G_trail", "G_trail_nodesc"):
            tot = 0.0
            for cond in df.cond.unique():
                if cond.startswith("no_") and cond != "no_altitude":
                    d = df[(df.cond == cond) & (df.kind == kind)].set_index(["seed", "kind"])
                    tot += float(np.nanmean(base.loc[d.index, "e_" + e].values - d["e_" + e].values))
            num(f"attrSum{e.replace('_', '')}{kind.replace('_', '')}", tot, 1, True)
        for e in ("G_trail", "ASCENT", "G_flat", "G_trail_nodesc"):
            num(f"attrAll{e.replace('_', '')}{kind.replace('_', '')}", float(base[base.index.get_level_values(1) == kind]["e_" + e].mean()), 1, True)


FR_EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT", "ASCENT_gh"]
FR_TAG = {"G_flat": "Gflat", "G_trail": "Gtrail", "G_trail_nodesc": "Gnodesc", "G_trail_hrgap": "Ghrgap",
          "ASCENT": "ASCENT", "ASCENT_gh": "ASCENTgh"}
FR_BIN = {"8-15": "A", "15-25": "B", "25-40": "C", ">40": "D"}


def tab_fitrec():
    fp = os.path.join(RES, "fitrec_dose_summary.csv")
    if not os.path.exists(fp):
        return
    t = pd.read_csv(fp)
    fx = pd.read_csv(os.path.join(RES, "fitrec_effects.csv")).set_index("estimator")
    nz = pd.read_csv(os.path.join(RES, "fitrec_noise.csv"), index_col=0)
    sel = pd.read_pickle(os.path.join(RES, "fitrec_selected.pkl"))
    est = pd.read_pickle(os.path.join(RES, "fitrec_estimates.pkl"))
    num("nFitrecRuns", f"{len(sel):,}".replace(",", "\\,"))
    num("nFitrecRunners", str(sel.userId.nunique()))
    for c in ("flat", "mid", "hilly"):
        num(f"nFitrec{c.capitalize()}", f"{int((sel.cls == c).sum()):,}".replace(",", "\\,"))
    num("nFitrecSteep", f"{int((sel.c >= 40).sum()):,}".replace(",", "\\,"))
    num("fitrecHrmaxMed", float(sel.groupby("userId").hr_max_u.first().median()), 0)
    num("fitrecRunsPerRunner", float(sel.groupby("userId").size().median()), 0)
    num("fitrecDurMed", float(sel.dur.median() / 60), 0)
    hdr = ["Classic", "Trail", "No-descent", "HR-GAP", "ASCENT", "ASCENT$^{\\dagger}$"]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(FR_EST) + "@{}}", r"\toprule",
             r"Climbing (m\,km$^{-1}$) & " + " & ".join(hdr) + r" \\", r"\midrule"]
    for b in ["8-15", "15-25", "25-40", ">40"]:
        cells = []
        for e in FR_EST:
            r = t[(t.bin == b) & (t.estimator == e)].iloc[0]
            cells.append(f"{f(r['trim'])} ({f(r['median'])})")
            bt = FR_BIN[b]
            num(f"frTrim{bt}{FR_TAG[e]}", float(r["trim"]), 1, True)
            num(f"frMed{bt}{FR_TAG[e]}", float(r["median"]), 1, True)
            num(f"frLo{bt}{FR_TAG[e]}", float(r["lo"]), 1, True)
            num(f"frHi{bt}{FR_TAG[e]}", float(r["hi"]), 1, True)
            num(f"frN{bt}{FR_TAG[e]}", str(int(r["n_users"])))
        lab = b.replace(">", "$>$").replace("-", "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    cells = []
    for e in FR_EST:
        cells.append(f"{f(fx.loc[e, 'slope10'])}")
        num(f"frSlope{FR_TAG[e]}", float(fx.loc[e, "slope10"]), 2, True)
        num(f"frSlopeLo{FR_TAG[e]}", float(fx.loc[e, "slope_lo"]), 2, True)
        num(f"frSlopeHi{FR_TAG[e]}", float(fx.loc[e, "slope_hi"]), 2, True)
        num(f"frPerHour{FR_TAG[e]}", float(fx.loc[e, "per_hour"]), 1, True)
        num(f"frPerHourLo{FR_TAG[e]}", float(fx.loc[e, "per_hour_lo"]), 1, True)
        num(f"frPerHourHi{FR_TAG[e]}", float(fx.loc[e, "per_hour_hi"]), 1, True)
        num(f"frGapHilly{FR_TAG[e]}", float(fx.loc[e, "gap"]), 1, True)
        num(f"frGapHillyLo{FR_TAG[e]}", float(fx.loc[e, "gap_lo"]), 1, True)
        num(f"frGapHillyHi{FR_TAG[e]}", float(fx.loc[e, "gap_hi"]), 1, True)
        num(f"frGapHillyMed{FR_TAG[e]}", float(fx.loc[e, "gap_median"]), 1, True)
        for c in ("flat", "mid", "hilly"):
            num(f"frCov{c.capitalize()}{FR_TAG[e]}", float(fx.loc[e, "cov_" + c]), 0)
        num(f"frNoiseSd{FR_TAG[e]}", float(nz.loc[e, "control_sd"]), 1)
        num(f"frSessSd{FR_TAG[e]}", float(nz.loc[e, "session_sd"]), 1)
        num(f"frMeanFlat{FR_TAG[e]}", float(fx.loc[e, "mean_flat"]), 1)
    lines.append(r"Slope per 10 m\,km$^{-1}$ & " + " & ".join(cells) + r" \\")
    lines.append(r"\quad 95\,\% CI & " + " & ".join(f"[{f(fx.loc[e, 'slope_lo'])}, {f(fx.loc[e, 'slope_hi'])}]" for e in FR_EST) + r" \\")
    lines.append(r"Duration effect per hour & " + " & ".join(f(fx.loc[e, "per_hour"]) for e in FR_EST) + r" \\")
    lines.append(r"Valid runs, $\geq 25$ m\,km$^{-1}$ (\%) & " + " & ".join(f"{fx.loc[e, 'cov_hilly']:.0f}" for e in FR_EST) + r" \\")
    lines.append(r"Session SD within runner, flat runs & " + " & ".join(f"{nz.loc[e, 'session_sd']:.1f}" for e in FR_EST) + r" \\")
    lines.append(r"Flat-vs-flat control SD & " + " & ".join(f"{nz.loc[e, 'control_sd']:.1f}" for e in FR_EST) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    for ln in lines:
        assert "%" not in ln.replace(r"\%", ""), ln
    open(os.path.join(TAB, "tab_fitrec.tex"), "w").write("\n".join(lines))
    num("nFitrecGapUsers", str(int(fx.loc["ASCENT", "n_users"])))
    num("nFitrecSlopeUsers", str(int(fx.loc["ASCENT", "n_users_slope"])))
    # 5-95 % range of per-runner gaps (hilly minus flat) for the dispersion statement
    for e in ("G_trail_hrgap", "ASCENT", "G_trail"):
        pg = pd.read_pickle(os.path.join(RES, f"fitrec_gaps_{e}.pkl"))
        num(f"frGapPZeroFive{FR_TAG[e]}", float(pg.gap.quantile(0.05)), 1, True)
        num(f"frGapPNineFive{FR_TAG[e]}", float(pg.gap.quantile(0.95)), 1, True)
    # ASCENT internals by class
    est["bin"] = np.where(est.c <= 8, "flat", np.where(est.c >= 40, "steep", "other"))
    g = est.groupby("bin")
    num("frTauFlat", float(g.ASCENT_tau.mean()["flat"]), 0)
    num("frTauSteep", float(g.ASCENT_tau.mean()["steep"]), 0)
    num("frDriftFlat", float(g.ASCENT_drift.mean()["flat"] * 100), 1)
    num("frDriftSteep", float(g.ASCENT_drift.mean()["steep"] * 100), 1)
    num("frShareClimbSteep", float(g.share_climb.mean()["steep"] * 100), 0)
    num("frShareDescSteep", float(g.share_descent.mean()["steep"] * 100), 0)
    num("frDurSteep", float(g.dur.mean()["steep"] / 60), 0)
    num("frDurFlat", float(g.dur.mean()["flat"] / 60), 0)
    # diagnostics on the steepest runs (variants), if present
    fp = os.path.join(RES, "fitrec_diag_gaps.pkl")
    if os.path.exists(fp):
        dr = pd.read_pickle(fp)
        u = dr.groupby("userId").mean(numeric_only=True)
        from scipy import stats as _st
        for v, tag in (("base", "Base"), ("trail_h", "TrailH"), ("gait_h", "GaitH"), ("no_lag", "NoLag"), ("no_drift", "NoDrift"),
                       ("climbs_only", "Climbs"), ("flats_only", "Flats"), ("no_descents", "NoDesc"),
                       ("climb_minus_flat_within", "Within")):
            x = u[v].dropna()
            m, lo, hi = float(x.mean()), np.nan, np.nan
            if len(x) > 2:
                hh = _st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                lo, hi = m - hh, m + hh
            num(f"frDiagMed{tag}", float(x.median()), 1, True)
            num(f"frDiagTrim{tag}", float(_st.trim_mean(x, 0.1)), 1, True)
            num(f"frDiagLo{tag}", lo, 1, True)
            num(f"frDiagHi{tag}", hi, 1, True)
            num(f"frDiagN{tag}", str(len(x)))
        ds = pd.read_pickle(os.path.join(RES, "fitrec_diag_steep.pkl"))
        num("frTauCapSteep", float((ds[ds.role == "steep"].tau >= 120).mean() * 100), 0)
        num("frTauCapFlat", float((ds[ds.role == "flat"].tau >= 120).mean() * 100), 0)


def tab_v2():
    """ASCENT v2: dev candidates, test half, simulation replay and case study."""
    fp = os.path.join(RES, "fitrec_v2_test_summary.csv")
    if not os.path.exists(fp):
        return
    bins = ["8-15", "15-25", "25-40", ">40"]
    bt = {"8-15": "A", "15-25": "B", "25-40": "C", ">40": "D"}
    # development candidates
    dev = pd.read_csv(os.path.join(RES, "fitrec_v2_dev_summary.csv")).set_index("candidate")
    cand = [("v1", "v1"), ("T", "lag fixed at 55 s"), ("TP", "lag prior tightened (sd 15 s, cap 80 s)"),
            ("A", "surface `auto'"), ("TA", "lag fixed + `auto'"), ("TAL75", "lag fixed + `auto' + $\\lambda=0.75$"),
            ("TAL100", "lag fixed + `auto' + $\\lambda=1$"), ("TAh", "lag fixed + `auto' + Huber $c=1.2$"),
            ("TAD", "lag fixed + `auto' + drift sd 0.03/h (= v2; $\\lambda=0.5$)"),
            ("TAnd", "lag fixed + `auto', no drift (diagnostic)"), ("TAW", "lag fixed + `auto', no weights (diagnostic)")]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Candidate & 25--40 & $>$40 & Session SD & Valid hilly (\%) & Level \\", r"\midrule"]
    for k, lab in cand:
        if k not in dev.index:
            continue
        r = dev.loc[k]
        lines.append(f"{lab} & {f(r['gap25-40'])} & {f(r['gap>40'])} & {r['sess_sd_flat']:.2f} & {r['cov_hilly']:.0f} & {r['mean_flat']:.1f} \\\\")
        kt = {"v1": "One", "TAL75": "TALsf", "TAL100": "TALone"}.get(k, k)
        for b in ("25-40", ">40"):
            num(f"vdev{bt[b]}{kt}", float(r[f"gap{b}"]), 1, True)
        num(f"vdevSd{kt}", float(r["sess_sd_flat"]), 2)
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_v2dev.tex"), "w").write("\n".join(lines))
    num("nVdevRunners", str(int(pd.read_pickle(os.path.join(RES, "fitrec_v2_dev.pkl")).userId.nunique())))
    num("nVdevSteep", str(int(dev.loc["v1", "n>40"])))
    num("nVdevHilly", str(int(dev.loc["v1", "n25-40"])))
    # test half
    tst = pd.read_csv(fp).set_index("candidate")
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Estimator & 8--15 & 15--25 & 25--40 & $>$40 & Sess.\ SD & Valid (\%) \\", r"\midrule"]
    for k, lab in (("v1", "\\ascent{} v1"), ("TAD", "\\ascent{} v2")):
        r = tst.loc[k]
        cells = [f"{f(r[f'gap{b}'])} [{f(r[f'lo{b}'])}, {f(r[f'hi{b}'])}]" for b in bins]
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {r['sess_sd_flat']:.2f} & {r['cov_hilly']:.0f} \\\\")
        tag = "One" if k == "v1" else "Two"
        for b in bins:
            num(f"vtest{bt[b]}{tag}", float(r[f"gap{b}"]), 1, True)
            num(f"vtestLo{bt[b]}{tag}", float(r[f"lo{b}"]), 1, True)
            num(f"vtestHi{bt[b]}{tag}", float(r[f"hi{b}"]), 1, True)
            num(f"vtestN{bt[b]}{tag}", str(int(r[f"n{b}"])))
        num(f"vtestSd{tag}", float(r["sess_sd_flat"]), 2)
        num(f"vtestCov{tag}", float(r["cov_hilly"]), 0)
    # the Garmin-like reconstructions on the same test half (from the per-run gaps of the full test)
    dr = pd.read_pickle(os.path.join(RES, "fitrec_dose.pkl"))
    dr = dr[dr.userId % 2 == 1]
    ez = pd.read_pickle(os.path.join(RES, "fitrec_estimates.pkl"))
    ez = ez[ez.userId % 2 == 1]
    from scipy import stats as _st
    for e, tag, lab in (("G_trail_hrgap", "Ghrgap", "HR-GAP"), ("G_trail_nodesc", "Gnodesc", "No-descent"),
                        ("G_flat", "Gflat", "Garmin-like classic"), ("G_trail", "Gtrail", "Garmin-like trail")):
        cells = []
        for b in bins:
            u = dr[dr.bin == b].groupby("userId")[e].mean().dropna()
            m, lo, hi = _st.trim_mean(u, 0.1), np.nan, np.nan
            if len(u) > 2:
                hh = _st.t.ppf(0.975, len(u) - 1) * u.std(ddof=1) / np.sqrt(len(u))
                lo, hi = u.mean() - hh, u.mean() + hh
            num(f"vtest{bt[b]}{tag}", float(m) if len(u) > 5 else np.nan, 1, True)
            num(f"vtestLo{bt[b]}{tag}", lo, 1, True)
            num(f"vtestHi{bt[b]}{tag}", hi, 1, True)
            num(f"vtestN{bt[b]}{tag}", str(len(u)))
            cells.append(f"{f(m)} [{f(lo)}, {f(hi)}]")
        sd = float(ez[ez.cls == "flat"].groupby("userId")[e].std().median())
        cov = float(ez[ez.cls == "hilly"][e].notna().mean() * 100)
        num(f"vtestSd{tag}", sd, 2)
        num(f"vtestCov{tag}", cov, 0)
        if tag in ("Ghrgap", "Gnodesc"):
            lines.append(f"{lab} & " + " & ".join(cells) + f" & {sd:.2f} & {cov:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_v2test.tex"), "w").write("\n".join(lines))
    num("nVtestRunners", str(int(pd.read_pickle(os.path.join(RES, "fitrec_v2_test.pkl")).userId.nunique())))
    # paired per-runner difference v2 minus HR-GAP on the test half
    # paired gaps per run from the v2 test file (same runs for both estimators)
    t2 = pd.read_pickle(os.path.join(RES, "fitrec_v2_test.pkl"))
    t2 = t2.merge(ez[["id", "G_trail_hrgap"]], on="id", how="left")
    W = 30 * 86400
    rows = []
    for uid, g in t2.groupby("userId"):
        flat = g[g.cls == "flat"]
        for _, r in g[g.c > 8].iterrows():
            near = flat[np.abs(flat.start_ts - r.start_ts) <= W]
            if len(near) == 0 or not (np.isfinite(r["TAD"]) and np.isfinite(r["G_trail_hrgap"])):
                continue
            n2, nh = near["TAD"].dropna(), near["G_trail_hrgap"].dropna()
            if len(n2) == 0 or len(nh) == 0:
                continue
            rows.append(dict(userId=uid, bin=bins[min(np.searchsorted([8, 15, 25, 40], r.c, side="right") - 1, 3)],
                             d=(r["TAD"] - n2.mean()) - (r["G_trail_hrgap"] - nh.mean())))
    pr = pd.DataFrame(rows)
    for b in bins:
        u = pr[pr.bin == b].groupby("userId").d.mean()
        hh = _st.t.ppf(0.975, len(u) - 1) * u.std(ddof=1) / np.sqrt(len(u)) if len(u) > 2 else np.nan
        num(f"vpair{bt[b]}", float(u.mean()), 1, True)
        num(f"vpairLo{bt[b]}", float(u.mean() - hh), 1, True)
        num(f"vpairHi{bt[b]}", float(u.mean() + hh), 1, True)
        num(f"vpairN{bt[b]}", str(len(u)))
    # simulation replay
    fp2 = os.path.join(RES, "main_v2.pkl")
    if os.path.exists(fp2):
        m = pd.read_pickle(fp2)
        lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                 r"Course & HR-GAP & \ascent{} v1 & \ascent{} v2 & \ascent{} v2, surface `auto' \\", r"\midrule"]
        for k in ORDER:
            d = m[m.kind == k]
            cells = []
            for e, tag in (("G_trail_hrgap", "Ghrgap"), ("ASCENT", "One"), ("ASCENT2", "Two"), ("ASCENT2auto", "TwoAuto")):
                err = (d[e] - d["target"]).dropna()
                cells.append(f"{f(err.mean())} [{np.sqrt(np.mean(err**2)):.1f}]")
                num(f"vsim{k.replace('_', '')}{tag}", float(err.mean()), 1, True)
                num(f"vsimR{k.replace('_', '')}{tag}", float(np.sqrt(np.mean(err**2))), 1)
            lines.append(f"{LABEL[k]} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "tab_v2sim.tex"), "w").write("\n".join(lines))
        z = (m["ASCENT2"] - m["target"]) / m["ASCENT2_sd"]
        num("vsimCovOne", float(np.nanmean(np.abs(z) < 1) * 100), 0)
        num("vsimCovTwo", float(np.nanmean(np.abs(z) < 2) * 100), 0)
        num("vsimHauto", float(m[m.kind == "road_hilly"]["ASCENT2auto_h"].mean()), 3)
    fp3 = os.path.join(RES, "mismatch_v2.pkl")
    if os.path.exists(fp3):
        mm = pd.read_pickle(fp3)
        for w in mm.world.drop_duplicates():
            for k in ("mountain", "skyrun", "trail_runnable"):
                d = mm[(mm.world == w) & (mm.kind == k)]
                c = d.dropna(subset=["G_trail_hrgap", "ASCENT2"])
                for e, tag in (("ASCENT2", "Two"), ("ASCENT", "One"), ("G_trail_hrgap", "Ghrgap")):
                    err = c[e] - c["target"]
                    num(f"vmm{WORLDTAG[w]}{k.replace('_', '')}{tag}", float(err.mean()), 1, True)
                    num(f"vmmR{WORLDTAG[w]}{k.replace('_', '')}{tag}", float(np.sqrt(np.mean(err**2))), 1)
    # case study
    fp4 = os.path.join(RES, "real_effects_v2.csv")
    if os.path.exists(fp4):
        fx = pd.read_csv(fp4)
        for per, pt in (("all", "All"), ("recent", "Recent")):
            for e, tag in (("ASCENT", "One"), ("ASCENT2", "Two"), ("ASCENT2auto", "TwoAuto")):
                r = fx[(fx.period == per) & (fx.estimator == e)].iloc[0]
                num(f"vreal{pt}{tag}", float(r["gap"]), 1, True)
                num(f"vrealLo{pt}{tag}", float(r["lo"]), 1, True)
                num(f"vrealHi{pt}{tag}", float(r["hi"]), 1, True)
                num(f"vrealRace{pt}{tag}", float(r["race_gap"]), 1, True)
                num(f"vrealSd{pt}{tag}", float(r["sd_train"]), 1)
                num(f"vrealRoad{pt}{tag}", float(r["mean_road"]), 1)
        de = pd.read_csv(os.path.join(RES, "real_estimates_v2.csv")).sort_values("date")
        num("vrealTrackedTwo", float(de.disp_ASCENT2_race.iloc[-1]), 1)
        num("vrealTrackedOne", float(de.disp_ASCENT_race.iloc[-1]), 1)
        num("vrealRoadLoopClimb", float(de[de.surface == "road"].dplus_per_km.median()), 0)


def tab_real_extra():
    if not os.path.exists(os.path.join(RES, "real_consistency.csv")):
        return
    c = pd.read_csv(os.path.join(RES, "real_consistency.csv"), dtype={"id": str})
    both = c[(c.climb_n >= 8) & (c.flat_n >= 8)].dropna(subset=["climb", "flat"])
    dlt = both["climb"] - both["flat"]
    num("consMean", float(dlt.mean()), 1, True)
    num("consLoaLo", float(dlt.mean() - 1.96 * dlt.std()), 1, True)
    num("consLoaHi", float(dlt.mean() + 1.96 * dlt.std()), 1, True)
    num("consWorst", float(dlt.loc[dlt.abs().idxmax()]), 1, True)
    num("consSD", float(dlt.std()), 1)
    num("consN", str(len(both)))
    se = dlt.std() / np.sqrt(len(both))
    num("consLo", float(dlt.mean() - 1.96 * se), 1, True)
    num("consHi", float(dlt.mean() + 1.96 * se), 1, True)
    tr = c[(c.surface == "trail") & ~c.race]
    num("hFreeMedianTrail", float(np.nanmedian(tr["h_free"])), 2)
    a = pd.read_csv(os.path.join(RES, "real_estimates_hrmax170.csv"), dtype={"id": str})
    a = a.merge(c[["id", "ascent_nodrift"]], on="id")
    rec = a[a.date >= "2026-06-01"]
    for g, gt in (("road", "Road"), ("trail", "Trail")):
        x = rec[(rec.surface == g) & ~rec.race]
        num(f"noDrift{gt}", float(x["ascent_nodrift"].mean()), 1)
        num(f"withDrift{gt}", float(x["ASCENT"].mean()), 1)
    num("noDriftGap", float(rec[(rec.surface == "trail") & ~rec.race]["ascent_nodrift"].mean()
                            - rec[(rec.surface == "road") & ~rec.race]["ascent_nodrift"].mean()), 1, True)
    num("tauMedianReal", float(np.nanmedian(a["ASCENT_tau"])), 0)
    num("tauAtCap", float(np.mean(a["ASCENT_tau"] >= 120) * 100), 0)
    num("driftMedianReal", float(np.nanmedian(a["ASCENT_drift"])), 2)
    num("driftMedianRace", float(np.nanmedian(a[a.race]["ASCENT_drift"])), 2)
    num("driftMedianTrain", float(np.nanmedian(a[~a.race]["ASCENT_drift"])), 2)
    # display changes after the two 2026 night races
    a = a.sort_values("date").reset_index(drop=True)
    for rid, tag in (("19659497712", "Albondon"), ("20342007439", "Dehesa"), ("19750627336", "Trevelez")):
        k = int(np.where(a["id"] == rid)[0][0])
        for col, ct in (("disp_G_trail", "Garmin"), ("disp_ASCENT_race", "Race"), ("disp_ASCENT_kalman", "Kalman")):
            num(f"dispChg{tag}{ct}", float(a.loc[k, col] - a.loc[k - 1, col]), 1, True)
    num("hrMaxObs", float(a["hr_p99"].max()), 0)
    r = pd.read_csv(os.path.join(RES, "real_robustness.csv"), dtype={"id": str})
    rng_tau = r[["tau20", "tau55", "tau120"]].max(axis=1) - r[["tau20", "tau55", "tau120"]].min(axis=1)
    rng_tau2 = r[["tau20", "tau55", "tau120", "tau240"]].max(axis=1) - r[["tau20", "tau55", "tau120", "tau240"]].min(axis=1)
    num("tauRangeMedian", float(np.nanmedian(rng_tau)), 1)
    num("tauRangeMax", float(np.nanmax(rng_tau)), 1)
    num("tauRangePnine", float(np.nanpercentile(rng_tau, 90)), 1)
    num("tauRangeMedianWide", float(np.nanmedian(rng_tau2)), 1)
    num("tauRangeMaxWide", float(np.nanmax(rng_tau2)), 1)
    rr = r[r.date >= "2026-06-01"]
    for col, tag in (("exdesc", "ExDesc"), ("first45", "First")):
        tr = rr[(~rr.race)][col].mean()
        rc = rr[rr.race & (rr.surface == "trail")][col].mean()
        num(f"raceGap{tag}", float(rc - tr), 1, True)
    base_tr = a[(a.date >= "2026-06-01") & ~a.race]["ASCENT"].mean()
    base_rc = a[(a.date >= "2026-06-01") & a.race & (a.surface == "trail")]["ASCENT"].mean()
    num("raceGapBase", float(base_rc - base_tr), 1, True)


def write_numbers():
    lines = ["% generated by make_tables.py"]
    for k, v in sorted(NUM.items()):
        assert k.isalpha(), k
        lines.append(rf"\newcommand{{\n{k}}}{{{v}}}")
    open(os.path.join(ROOT, "report", "numbers.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    tab_main()
    tab_ablation()
    tab_sens()
    tab_real()
    sens_macros()
    tab_mismatch()
    tab_long()
    tab_attr()
    tab_real_extra()
    tab_fitrec()
    tab_v2()
    extra = sys.argv[1:]
    write_numbers()
    print(len(NUM), "numbers written")
