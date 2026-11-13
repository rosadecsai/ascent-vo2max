"""Tables and number macros added in the first revision (Sports Engineering).

Reads results/revision/gaps.csv and summary.json (fitrec_revision_analysis.py), results/sim_sampling.pkl
(sim_sampling.py), results/attribution.pkl, results/fitrec_effects.csv and fitrec_noise.csv.
Writes report/tables/tab_r1_*.tex and report/numbers_rev.tex.  The macros of the original
submission (report/numbers.tex, make_tables.py) are left untouched.
"""
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
REV = os.path.join(RES, "revision")
TAB = os.path.join(ROOT, "report", "tables")
NUM = {}
LABELS = ["8-15", "15-25", "25-40", ">40"]
BT = {"8-15": "A", "15-25": "B", "25-40": "C", ">40": "D"}
BL = {"8-15": "8--15", "15-25": "15--25", "25-40": "25--40", ">40": "$>$40"}
RELEASE, RELEASE_SUBMITTED = "v1.1.0", "v1.0.0"
DELTA = 3.5


def f(x, nd=1, sign=True):
    """Number for a table cell or the text, with a proper minus sign and no negative zero."""
    if x is None or not np.isfinite(x):
        return "n/a"
    x = round(float(x), nd)
    if x == 0:
        return f"{0:.{nd}f}"
    s = f"{abs(x):.{nd}f}"
    if x < 0:
        return "$-$" + s
    return ("+" + s) if sign else s


def num(name, val):
    assert name.isalpha(), name
    assert name not in NUM, name
    NUM[name] = str(val)


def ci(lo, hi, nd=1):
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def write(name, lines):
    for ln in lines:
        assert "%" not in ln.replace(r"\%", ""), ln
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")


SETTAG = {"all": "All", "common": "Common", "matched": "Matched", "matched_test": "MatchedTest", "anchor_base": "AnchorBase",
          "anchor_rest45": "AnchorRestLo", "anchor_rest65": "AnchorRestHi", "anchor_maxm10": "AnchorMaxLo",
          "anchor_maxp10": "AnchorMaxHi", "test": "Test", "dev": "Dev", "allr": "Allr", "spacing_fine": "SpacingFine",
          "spacing_coarse": "SpacingCoarse", "paired_test": "PairedTest", "win15": "WinFifteen", "win60": "WinSixty",
          "win90": "WinNinety"}
ESTTAG = {"G_flat": "Gflat", "G_trail": "Gtrail", "G_trail_nodesc": "Gnodesc", "G_trail_hrgap": "Ghrgap", "ASCENT": "VOne",
          "ASCENT_gh": "VOneGh", "T_road": "TRoad", "D_road": "DRoad", "v2_road": "VTwoRoad", "v2_trail": "VTwoTrail", "v2": "VTwo"}


def esttag(e):
    e = re.sub(r"_(rest45|rest65|maxm10|maxp10)$", "", e)
    if "-" in e:
        a, b = e.split("-")
        return ESTTAG[a] + "Minus" + ESTTAG[b]
    return ESTTAG[e]


def load():
    T = pd.read_csv(os.path.join(REV, "gaps.csv"))
    J = json.load(open(os.path.join(REV, "summary.json")))
    T[["set", "est"]] = T.label.str.split("|", expand=True)
    return T, J


def cell(T, s, e, b, n=False):
    r = T[(T.set == s) & (T.est == e) & (T.bin == b)]
    if len(r) == 0:
        return "n/a"
    r = r.iloc[0]
    out = f"{f(r.trim)} {ci(r.lo, r.hi)}"
    if n:
        out += f" ({int(r.n_runners)})"
    return out


def gap_macros(T):
    for _, r in T.iterrows():
        base = "g" + SETTAG[r.set] + esttag(r.est) + BT[r.bin]
        num(base, f(r.trim))
        num(base + "ci", f"{f(r.lo)} to {f(r.hi)}")
        num(base + "n", int(r.n_runners))
        if r.trim < 0 and r.hi < 0:      # magnitude form ("x lower than"), used in the abstract and conclusions
            num(base + "abs", f(-r.trim, sign=False))
            num(base + "ciabs", f"{f(-r.hi, sign=False)} to {f(-r.lo, sign=False)}")
        elif r.trim < 0:
            num(base + "abs", f(-r.trim, sign=False))


# ------------------------------------------------------------------------------------------
def tab_fitrec(T, J):
    fx = pd.read_csv(os.path.join(RES, "fitrec_effects.csv")).set_index("estimator")
    rows = [("G_flat", "Classic reconstruction"), ("G_trail", "Trail reconstruction"), ("G_trail_nodesc", "No-descent reconstruction"),
            ("G_trail_hrgap", "HR-GAP reconstruction"), ("ASCENT_gh", "\\ascent{} v1, walking labels"),
            ("ASCENT", "\\ascent{} v1, no gait labels")]
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Climbing class (\mkm)} & & \\ \cmidrule(lr){2-5}",
         r"Estimator & 8--15 & 15--25 & 25--40 & $>$40 & Slope per 10 \mkm & Valid (\%) \\", r"\midrule"]
    for e, lab in rows:
        cells = [cell(T, "all", e, b) for b in LABELS]
        L.append(f"{lab} & " + " & ".join(cells) + f" & {f(fx.loc[e, 'slope10'], 2)} {ci(fx.loc[e, 'slope_lo'], fx.loc[e, 'slope_hi'], 2)}"
                 + f" & {fx.loc[e, 'cov_hilly']:.0f} \\\\")
        ns = []
        for b in LABELS:
            r = T[(T.set == "all") & (T.est == e) & (T.bin == b)].iloc[0]
            ns.append(f"{int(r.n_runners)} ({int(r.n_runs)})")
        L.append(r"\quad runners (runs) & " + " & ".join(ns) + f" & {int(fx.loc[e, 'n_users_slope'])} & \\\\")
        if e != rows[-1][0]:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_fitrec", L)


def tab_robust(T, J):
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Classic reconstruction} & \multicolumn{2}{c}{HR-GAP reconstruction} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Analysis & 25--40 & $>$40 & 25--40 & $>$40 \\", r"\midrule"]
    spec = [("All runs with an estimate (Table~\\ref{tab:fitrec})", "all", ""),
            ("Common set: runs valid for all five methods", "common", ""),
            ("Comparators of similar duration and intensity", "matched", ""),
            ("Comparators within $\\pm$15 days (instead of $\\pm$30)", "win15", ""),
            ("Comparators within $\\pm$60 days", "win60", ""),
            ("Comparators within $\\pm$90 days", "win90", ""),
            (None, None, None),
            ("\\HRrest{} 45 (instead of 55)", "anchor_rest45", "_rest45"),
            ("\\HRrest{} 65", "anchor_rest65", "_rest65"),
            ("\\HRmax{} lowered by 10", "anchor_maxm10", "_maxm10"),
            ("\\HRmax{} raised by 10", "anchor_maxp10", "_maxp10"),
            (None, None, None),
            ("Sample spacing up to 7 s", "spacing_fine", ""),
            ("Sample spacing above 7 s", "spacing_coarse", "")]
    for lab, s, suf in spec:
        if lab is None:
            L.append(r"\addlinespace[2pt]")
            continue
        cells = [cell(T, s, e, b, n=True) for e in ("G_flat", "G_trail_hrgap") for b in ("25-40", ">40")]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_robust", L)


V2ROWS = [("ASCENT", "\\ascent{} v1 (no gait labels)"), ("T_road", "\\quad HR lag fixed at 55 s only"),
          ("D_road", "\\quad tighter drift prior only"), ("v2_road", "\\quad both: v2 with a constant road prior"),
          ("v2_trail", "\\quad both, with a constant trail prior"), ("v2", "\\ascent{} v2 (both + climbing prior)"),
          ("G_trail_hrgap", "HR-GAP reconstruction"), ("G_trail_nodesc", "No-descent reconstruction")]


def tab_v2(T, J):
    sd = J["session_sd"]["test"]
    val = J["valid_hilly"]["test"]
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Climbing class (\mkm)} & & \\ \cmidrule(lr){2-5}",
         r"Estimator & 8--15 & 15--25 & 25--40 & $>$40 & Session SD & Valid (\%) \\", r"\midrule"]
    for e, lab in V2ROWS:
        if e == "G_trail_hrgap":
            L.append(r"\addlinespace[2pt]")
        cells = [cell(T, "test", e, b) for b in LABELS]
        L.append(f"{lab} & " + " & ".join(cells) + f" & {sd[e]:.2f} & {val[e]:.0f} \\\\")
    ns = [str(int(T[(T.set == "test") & (T.est == "v2") & (T.bin == b)].iloc[0].n_runners)) for b in LABELS]
    L += [r"\addlinespace[2pt]", r"Runners contributing (v2) & " + " & ".join(ns) + r" & " + str(sd["n"]) + r" & \\"]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_v2", L)


# ---- supplement ----------------------------------------------------------------------------
def tab_flow(J):
    L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Condition & Runs & Runners \\", r"\midrule"]
    names = {"runs with HR, GPS and altitude": "Running workouts with heart rate, GPS and altitude",
             "duration >= 25 min": "Duration of at least 25 min",
             "moving > 80 % of the time": "Moving more than 80\\,\\% of the time",
             "median HR > 100 bpm": "Median heart rate above 100 bpm",
             "mean sample spacing <= 30 s": "Mean sample spacing of at most 30 s",
             "mean altitude -100 to 3500 m": "Mean altitude between $-100$ and 3500 m$^{a}$",
             "climbing <= 150 m/km": "Climbing of at most 150 \\mkm$^{a}$",
             "gain/loss imbalance <= 0.5": "Imbalance between elevation gain and loss of at most 0.5$^{a}$",
             "altitude roughness <= 5 m": "Altitude roughness of at most 5 m$^{a}$",
             ">= 5 flat and >= 5 hilly runs per runner": "Runners with at least 5 flat and 5 hilly runs",
             "most recent 30 flat, 60 hilly, 20 intermediate": "Most recent 30 flat, 60 hilly and 20 intermediate runs per runner"}
    for r in J["flow"]:
        L.append(f"{names[r['step']]} & {r['runs']:,} & {r['runners']:,} \\\\".replace(",", "\\,"))
    L += [r"\bottomrule", r"\multicolumn{3}{@{}l}{\footnotesize $^{a}$ Altitude conditions added after a first pass (Section~\ref{s:chron}).}",
          r"\end{tabular}"]
    write("tab_r1_flow", L)


def tab_bins(J):
    L = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Class (\mkm) & Runs & Runners & Climbing (\mkm) & Duration (min) & Speed (m\,s$^{-1}$) & Intensity & Spacing (s) \\", r"\midrule"]
    lab = {"flat": "Flat, up to 8"} | BL
    for r in J["bins_descriptive"]:
        L.append(f"{lab[r['bin']]} & {r['runs']:,} & {r['runners']} & {r['c_med']:.0f} & {r['dur_med']:.0f} & {r['v_med']:.2f} & "
                 f"{r['int_med']:.2f} & {r['spacing_med']:.1f} \\\\".replace(",", "\\,"))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_bins", L)


ENAME = {"G_flat": "Classic", "G_trail": "Trail", "G_trail_nodesc": "No-descent", "G_trail_hrgap": "HR-GAP",
         "ASCENT": "\\ascent{} v1", "ASCENT_gh": "\\ascent{} v1, labels", "v2": "\\ascent{} v2"}


def tab_counts(T, J):
    cov = pd.DataFrame(J["coverage"])
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Estimator & Class (\mkm) & Runs & With estimate & (\%) & With comparator & Runners$^{a}$ \\", r"\midrule"]
    for e in ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT", "v2"]:
        for b in ["flat"] + LABELS:
            r = cov[(cov.estimator == e) & (cov.bin == b)].iloc[0]
            if b == "flat":
                L.append(f"{ENAME[e]} & Flat & {int(r.runs):,} & {int(r.valid):,} & {r.valid_pct:.0f} & -- & -- \\\\".replace(",", "\\,"))
            else:
                s = "all" if e != "v2" else "allr"
                t = T[(T.set == s) & (T.est == e) & (T.bin == b)]
                nr = int(t.iloc[0].n_runners) if len(t) else 0
                L.append(f" & {BL[b]} & {int(r.runs):,} & {int(r.valid):,} & {r.valid_pct:.0f} & {int(r.with_comparator):,} & {nr} \\\\".replace(",", "\\,"))
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\multicolumn{7}{@{}l}{\footnotesize $^{a}$ Runners with at least one run of the class that has an estimate and a flat comparator.}",
                  r"\end{tabular}"]
    write("tab_r1_counts", L)


def tab_matching(J):
    m = pd.DataFrame(J["matching"])
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Estimator & Class (\mkm) & Runs & Comparators per run & One comparator (\%) & Days to comparators \\", r"\midrule"]
    for e in ["G_flat", "ASCENT"]:
        for b in LABELS:
            r = m[(m.estimator == e) & (m.bin == b)].iloc[0]
            L.append(f"{ENAME[e] if b == LABELS[0] else ''} & {BL[b]} & {int(r.runs):,} & {r.ncomp_med:.0f} ({r.ncomp_q1:.0f}--{r.ncomp_q3:.0f}) & "
                     f"{r.one_comp_pct:.0f} & {r.dt_med:.0f} ({r.dt_q1:.0f}--{r.dt_q3:.0f}) \\\\".replace(",", "\\,"))
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_matching", L)


def tab_sets(T):
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Set of runs & Estimator & 8--15 & 15--25 & 25--40 & $>$40 \\", r"\midrule"]
    for s, lab in (("all", "All runs with an estimate"), ("common", "Common set"), ("matched", "Comparable comparators")):
        for k, e in enumerate(["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]):
            L.append(f"{lab if k == 0 else ''} & {ENAME[e]} & " + " & ".join(cell(T, s, e, b, n=True) for b in LABELS) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_sets", L)


def tab_anchors(T, J):
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Estimator & Anchors (bpm) & 8--15 & 15--25 & 25--40 & $>$40 & Flat level & Valid (\%) \\", r"\midrule"]
    an = [("anchor_base", "", "\\HRrest{} 55, \\HRmax{} as derived", "base"), ("anchor_rest45", "_rest45", "\\HRrest{} 45", "rest45"),
          ("anchor_rest65", "_rest65", "\\HRrest{} 65", "rest65"), ("anchor_maxm10", "_maxm10", "\\HRmax{} $-$10", "maxm10"),
          ("anchor_maxp10", "_maxp10", "\\HRmax{} $+$10", "maxp10")]
    for e in ["G_flat", "G_trail_hrgap", "v2"]:
        for k, (s, suf, lab, key) in enumerate(an):
            L.append(f"{ENAME[e] if k == 0 else ''} & {lab} & " + " & ".join(cell(T, s, e, b) for b in LABELS)
                     + f" & {J['anchor_levels'][e][key]:.1f} & {J['anchor_cov'][e][key]:.0f} \\\\")
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_anchors", L)


def tab_spacing(T, J):
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Sample spacing & Estimator & 8--15 & 15--25 & 25--40 & $>$40 \\", r"\midrule"]
    for s, lab in (("spacing_fine", f"Up to 7 s ({J['spacing']['fine_runs']:,} runs)".replace(",", "\\,")),
                   ("spacing_coarse", f"Above 7 s ({J['spacing']['coarse_runs']:,} runs)".replace(",", "\\,"))):
        for k, e in enumerate(["G_flat", "G_trail_hrgap", "ASCENT"]):
            L.append(f"{lab if k == 0 else ''} & {ENAME[e]} & " + " & ".join(cell(T, s, e, b, n=True) for b in LABELS) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_spacing", L)


def tab_decomp(T, J):
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Runners & Estimator & 8--15 & 15--25 & 25--40 & $>$40 \\", r"\midrule"]
    for s, lab in (("dev", f"Development half ({J['split']['dev_runners']})"), ("test", f"Test half ({J['split']['test_runners']})"),
                   ("allr", "All runners")):
        for k, (e, nm) in enumerate(V2ROWS):
            L.append(f"{lab if k == 0 else ''} & {nm} & " + " & ".join(cell(T, s, e, b, n=True) for b in LABELS) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_decomp", L)


def tab_equiv(T):
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Estimator & Quantity & 8--15 & 15--25 & 25--40 & $>$40 \\", r"\midrule"]
    out = {}
    for e, nm in (("v2", "\\ascent{} v2"), ("v2_road", "v2 with a constant road prior"), ("ASCENT", "\\ascent{} v1"),
                  ("G_trail_hrgap", "HR-GAP reconstruction"), ("G_trail_nodesc", "No-descent reconstruction"), ("G_flat", "Classic reconstruction")):
        r = T[(T.set == "test") & (T.est == e)].set_index("bin").loc[LABELS]
        L.append(f"{nm} & 90\\,\\% interval & " + " & ".join(ci(a, b) for a, b in zip(r.lo90, r.hi90)) + r" \\")
        marg = np.maximum(r.lo90.abs(), r.hi90.abs())
        L.append(" & smallest margin & " + " & ".join(f"{np.ceil(m * 10) / 10:.1f}" for m in marg) + r" \\")
        L.append(f" & equivalent at $\\pm${DELTA} & " + " & ".join("yes" if m <= DELTA else "no" for m in marg) + r" \\")
        L.append(r"\addlinespace[2pt]")
        out[e] = np.ceil(marg.values * 10) / 10
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_equiv", L)
    return out


def tab_simsamp():
    df = pd.read_pickle(os.path.join(RES, "sim_sampling.pkl"))
    est = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "v1", "v2"]
    nm = {"G_flat": "Classic", "G_trail": "Trail", "G_trail_nodesc": "No-descent", "G_trail_hrgap": "HR-GAP", "v1": "\\ascent{} v1",
          "v2": "\\ascent{} v2"}
    w = df.pivot_table(index=["seed", "cond"], columns="kind", values=est)
    res = {}
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Session minus flat road & Recording & " + " & ".join(nm[e] for e in est) + r" \\", r"\midrule"]
    for kind, klab in (("road_hilly", "Hilly road"), ("mountain", "Mountain trail")):
        for k, (cond, clab) in enumerate((("full", "Full"), ("nocad", "No cadence"), ("n500", "500 samples"))):
            cells = []
            for e in est:
                d = (w[(e, kind)] - w[(e, "road_flat")]).xs(cond, level="cond").dropna()
                res[(kind, cond, e)] = float(d.mean())
                cells.append(f(d.mean()))
            L.append(f"{klab if k == 0 else ''} & {clab} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    write("tab_r1_simsamp", L)
    num("rvSimSampN", df.seed.nunique())
    num("rvSimSampSpacing", f"{df[(df.cond == 'n500') & (df.kind == 'mountain')].spacing.median():.0f}")
    num("rvSimSampMaxDecim", f"{max(abs(res[(k, 'n500', e)] - res[(k, 'nocad', e)]) for k in ('road_hilly', 'mountain') for e in est):.1f}")
    for e, tag in (("G_flat", "Gflat"), ("G_trail_hrgap", "Ghrgap"), ("G_trail", "Gtrail"), ("v1", "VOne"), ("v2", "VTwo")):
        num(f"rvSimSamp{tag}Full", f(res[("mountain", "full", e)]))
        num(f"rvSimSamp{tag}Nocad", f(res[("mountain", "nocad", e)]))
        num(f"rvSimSamp{tag}Decim", f(res[("mountain", "n500", e)]))
    num("rvSimSampVTwoHillyNocad", f(res[("road_hilly", "nocad", "v2")]))
    num("rvSimSampVTwoHillyDecim", f(res[("road_hilly", "n500", "v2")]))


def attribution():
    a = pd.read_pickle(os.path.join(RES, "attribution.pkl"))
    a["e"] = a["G_trail"] - a["target"]
    x = a[a.kind == "mountain"]
    base = x[x.cond == "all"].set_index("seed")["e"]
    ideal = x[x.cond == "ideal"].set_index("seed")["e"]
    marg = {}
    for c in sorted(set(x.cond) - {"all", "ideal"}):
        d = x[x.cond == c].set_index("seed")["e"]
        marg[c.replace("no_", "")] = float(np.nanmean(base.loc[d.index].values - d.values))
    total = float(np.nanmean(base) - np.nanmean(ideal))
    s_all = sum(marg.values())
    num("rvAttrIdealGtrailmountain", f(np.nanmean(ideal)))
    num("rvAttrTotalGtrailmountain", f(total))
    num("rvAttrSumAllGtrailmountain", f(s_all))
    num("rvAttrAltGtrailmountain", f(marg["altitude"]))
    num("rvAttrInterGtrailmountain", f(total - s_all))
    small = [v for k, v in marg.items() if k not in ("ecc", "drift", "terrain", "gps_bias", "altitude")]
    num("rvAttrSmallMin", "0.0" if abs(min(small)) < 0.05 else f(min(small)))
    num("rvAttrSmallMax", f(max(small)))
    return marg, total


def misc(T, J, eq):
    p = J["participants"]
    num("rvPartMale", p["male"]); num("rvPartFemale", p["female"]); num("rvPartUnknown", p["unknown"])
    num("rvPartYearMin", p["year_min"]); num("rvPartYearMax", p["year_max"]); num("rvPartShareRecent", f"{p['share_2013_2015']:.0f}")
    num("rvPartRunsMed", f"{p['runs_med']:.0f}"); num("rvPartRunsIqr", f"{p['runs_q1']:.0f}--{p['runs_q3']:.0f}")
    num("rvPartSpanMed", f"{p['span_med_days']:.0f}")
    num("rvPartDurMed", f"{p['dur_med']:.0f}"); num("rvPartDurIqr", f"{p['dur_q1']:.0f}--{p['dur_q3']:.0f}")
    num("rvPartSpeedMed", f"{p['v_med']:.1f}"); num("rvPartSpeedIqr", f"{p['v_q1']:.1f}--{p['v_q3']:.1f}")
    num("rvPartSpacingMed", f"{p['spacing_med']:.0f}")
    num("rvPartHrmaxMed", f"{p['hrmax_med']:.0f}"); num("rvPartHrmaxIqr", f"{p['hrmax_q1']:.0f}--{p['hrmax_q3']:.0f}")
    num("rvPartLevelMed", f"{p['flat_level_classic']:.0f}")
    num("rvPartLevelIqr", f"{p['flat_level_classic_q1']:.0f}--{p['flat_level_classic_q3']:.0f}")
    num("rvPartOrienteering", p["orienteering"]); num("rvPartOrienteeringPct", f"{p['orienteering'] / p['n_runs'] * 100:.1f}")
    num("rvPartSpanIqr", f"{p['span_q1_days']:.0f}--{p['span_q3_days']:.0f}")
    d = J["diag"]
    num("rvDiagSteepRuns", d["steep_runs"]); num("rvDiagSteepRunners", d["steep_runners"])
    num("rvDiagSteepRunsTest", d["steep_runs_test"]); num("rvDiagSteepRunnersTest", d["steep_runners_test"])
    num("rvRelease", RELEASE); num("rvReleaseSubmitted", RELEASE_SUBMITTED)
    num("rvCommonPct", f"{J['common_runs']['pct']:.0f}"); num("rvCommonRuns", f"{J['common_runs']['n']:,}".replace(",", "\\,"))
    sh = []
    for e in ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]:
        for b in LABELS:
            a = T[(T.set == "all") & (T.est == e) & (T.bin == b)].trim.iloc[0]
            c = T[(T.set == "common") & (T.est == e) & (T.bin == b)].trim.iloc[0]
            sh.append(abs(a - c))
    num("rvCommonMaxShift", f"{max(sh):.1f}")
    m = J["matching_all"]
    num("rvMatchNcompMed", f"{m['ncomp_med']:.0f}"); num("rvMatchNcompIqr", f"{m['ncomp_q1']:.0f}--{m['ncomp_q3']:.0f}")
    num("rvMatchDtMed", f"{m['dt_med']:.0f}"); num("rvMatchDtIqr", f"{m['dt_q1']:.0f}--{m['dt_q3']:.0f}")
    num("rvMatchNoCompPct", f"{J['no_comparator']['pct']:.0f}")
    for days, tag in (("15", "Fifteen"), ("30", "Thirty"), ("60", "Sixty"), ("90", "Ninety")):
        num("rvWinPaired" + tag, f"{J['windows'][days]['pct_paired']:.0f}")
        num("rvWinSteepRunners" + tag, J["windows"][days]["runners_steep"])
    num("rvMatchPairedPct", f"{100 - J['no_comparator']['pct']:.0f}")
    num("rvMatchPairedRuns", f"{J['no_comparator']['paired']:,}".replace(",", "\\,"))
    num("rvMatchValidRuns", f"{J['no_comparator']['valid']:,}".replace(",", "\\,"))
    num("rvMatchNoCompRuns", f"{J['no_comparator']['valid'] - J['no_comparator']['paired']:,}".replace(",", "\\,"))
    # v2 on the test half
    v = T[(T.set == "test") & (T.est == "v2")].set_index("bin").loc[LABELS]
    num("rvVTwoTestMin", f(v.trim.min())); num("rvVTwoTestMax", f(v.trim.max()))
    for b, m_ in zip(LABELS, eq["v2"]):
        num("rvEqMargin" + BT[b], f"{m_:.1f}")
    num("rvEqMarginMax", f"{eq['v2'].max():.1f}")
    npass = int((eq["v2"] <= DELTA).sum())
    num("rvEqPassVTwo", {4: "all", 3: "three", 2: "two", 1: "one", 0: "none"}[npass])
    for e, tag in (("v2_road", "VTwoRoad"), ("G_trail_hrgap", "Ghrgap"), ("G_trail_nodesc", "Gnodesc")):
        num("rvEqPass" + tag, {4: "all", 3: "three", 2: "two", 1: "one", 0: "none"}[int((eq[e] <= DELTA).sum())])
        num("rvEqMarginMax" + tag, f"{eq[e].max():.1f}")
    s = J["session_sd"]["test"]
    num("rvSdVTwo", f"{s['v2']:.2f}"); num("rvSdGhrgap", f"{s['G_trail_hrgap']:.2f}"); num("rvSdVOne", f"{s['ASCENT']:.2f}")
    num("rvSdN", s["n"])
    num("rvSdDiffMed", f(s["v2-G_trail_hrgap_med"], 2)); num("rvSdDiffCi", f"{f(s['v2-G_trail_hrgap_lo'], 2)} to {f(s['v2-G_trail_hrgap_hi'], 2)}")
    pv = s["v2-G_trail_hrgap_p"]
    num("rvSdDiffP", f"= {pv:.3f}" if pv >= 0.001 else "$<$ 0.001")
    num("rvSdDiffFrac", f"{s['v2-G_trail_hrgap_frac']:.0f}")
    sa = J["session_sd"]["allr"]
    num("rvSdAllVTwo", f"{sa['v2']:.2f}"); num("rvSdAllGhrgap", f"{sa['G_trail_hrgap']:.2f}"); num("rvSdAllN", sa["n"])
    num("rvSdAllDiffMed", f(sa["v2-G_trail_hrgap_med"], 2))
    num("rvSdAllDiffCi", f"{f(sa['v2-G_trail_hrgap_lo'], 2)} to {f(sa['v2-G_trail_hrgap_hi'], 2)}")
    pa = sa["v2-G_trail_hrgap_p"]
    num("rvSdAllDiffP", f"= {pa:.3f}" if pa >= 0.001 else "$<$ 0.001")
    for sl in J["slopes"]:
        tag = ESTTAG[sl["estimator"]]
        num("rvSlopeInt" + tag, f(sl["slope_int"], 2)); num("rvSlopeIntCi" + tag, f"{f(sl['slope_int_lo'], 2)} to {f(sl['slope_int_hi'], 2)}")
    num("rvSpacingFineRuns", f"{J['spacing']['fine_runs']:,}".replace(",", "\\,"))
    num("rvSpacingCoarseRuns", f"{J['spacing']['coarse_runs']:,}".replace(",", "\\,"))
    r = J["runner_only_boot"]
    num("rvRunnerOnlyBoot", f"{f(r['lo'])} to {f(r['hi'])}")
    num("rvHmeanC", f"{J['split']['h_mean_by_bin']['25-40']:.3f}"); num("rvHmeanD", f"{J['split']['h_mean_by_bin']['>40']:.3f}")
    nm = J["nodesc_missing"]
    num("rvNodescMissN", nm["n"]); num("rvNodescMissC", f"{nm['c_med']:.0f}"); num("rvNodescKeptC", f"{nm['c_med_kept']:.0f}")
    num("rvNodescMissDur", f"{nm['dur_med']:.0f}"); num("rvNodescKeptDur", f"{nm['dur_med_kept']:.0f}")
    rm = J["recon_missing"]
    num("rvReconMissN", rm["n"]); num("rvReconMissC", f"{rm['c_med']:.0f}"); num("rvReconKeptC", f"{rm['c_med_kept']:.0f}")
    num("rvReconMissV", f"{rm['v_med']:.1f}"); num("rvReconKeptV", f"{rm['v_med_kept']:.1f}")
    num("rvReconMissInt", f"{rm['int_med']:.2f}"); num("rvReconKeptInt", f"{rm['int_med_kept']:.2f}")


def write_numbers():
    lines = ["% generated by make_tables_rev.py (first revision)"]
    for k, v in sorted(NUM.items()):
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    open(os.path.join(ROOT, "report", "numbers_rev.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    T, J = load()
    gap_macros(T)
    tab_fitrec(T, J)
    tab_robust(T, J)
    tab_v2(T, J)
    tab_flow(J)
    tab_bins(J)
    tab_counts(T, J)
    tab_matching(J)
    tab_sets(T)
    tab_anchors(T, J)
    tab_spacing(T, J)
    tab_decomp(T, J)
    eq = tab_equiv(T)
    tab_simsamp()
    marg, total = attribution()
    misc(T, J, eq)
    write_numbers()
    print(len(NUM), "numbers written;", "attribution:", {k: round(v, 2) for k, v in marg.items()}, "total", round(total, 2))
