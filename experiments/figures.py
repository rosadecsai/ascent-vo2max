"""Figures for the report (simulation and real data)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import plt, save, C, MARK, NAME, SHORT, INK, INK2, MUTED, GRID, TRUTH  # noqa: E402
from common import sm, es, SURF, LABEL  # noqa: E402
from ascent import physiology as ph  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
ORDER = ["road_flat", "road_hilly", "trail_runnable", "mountain", "skyrun"]


# ----------------------------------------------------------------------------------------
def fig_main():
    df = pd.read_pickle(os.path.join(RES, "main.pkl"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, race in zip(axes, (False, True)):
        d = df[df.race == race]
        for j, e in enumerate(EST):
            err = d[e] - d["target"]
            g = err.groupby(d["kind"])
            m, s = g.mean(), g.std()
            few = g.count() < 10                      # too few estimates: not shown
            m[few], s[few] = np.nan, np.nan
            y = np.arange(len(ORDER)) + (j - 2.0) * 0.15
            ax.errorbar(m[ORDER], y, xerr=s[ORDER], fmt=MARK[e], color=C[e], ms=4.2,
                        elinewidth=1.0, capsize=0, label=SHORT[e],
                        mec="white", mew=0.6, zorder=3)
        ax.axvline(0, color=INK2, lw=0.9, zorder=2)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LABEL[k] for k in ORDER])
        ax.set_xlabel("Error vs true VO$_2$max (ml·kg$^{-1}$·min$^{-1}$)")
        ax.set_title("Race sessions" if race else "Training sessions", loc="left")
        ax.set_xlim(-31, 6)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 1.08),
               handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, os.path.join(FIG, "fig_main_bias.pdf"))


def fig_example(seed=7, kind="mountain"):
    rng = np.random.default_rng(seed)
    r = sm.random_runner(rng, vo2max=50.0, hr_rest=52.0, hr_max=180.0, econ=1.0, up_eff=1.0,
                         walk_econ=1.0, ecc_lambda=0.7, drift=0.06)
    s = sm.random_session(r, kind, True, rng)
    p = es.prep(s.t, s.hr, s.dist, s.alt, s.cad, s.eta_hat)
    g = es.garmin_like(p, r.hr_rest, r.hr_max, "trail", return_windows=True)
    a = es.ascent(p, r.hr_rest, r.hr_max, SURF[kind], return_windows=True)
    W, GW = a["windows"], g["windows"]
    tmin = p["t"] / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 1.3, 1.5]))
    ax = axes[0]
    z = s.truth["z"]
    ax.plot(tmin, z, color=INK2, lw=1.2)
    zr = z.max() - z.min()
    lo = z.min() - 0.18 * zr
    desc = s.truth["grade"] < -0.05
    walk = s.truth["walk"]
    ax.fill_between(tmin, lo, lo + 0.07 * zr, where=desc, color="#898781", lw=0, step="mid",
                    label="descent (grade < −5 %)")
    ax.fill_between(tmin, lo + 0.08 * zr, lo + 0.15 * zr, where=walk, color="#6da7ec", lw=0,
                    step="mid", label="power-hiking")
    ax.set_ylim(lo - 0.02 * zr, z.max() + 0.08 * zr)
    ax.set_ylabel("Altitude (m)")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    ax.set_title(f"Simulated mountain race — true VO$_2$max {r.target:.1f}", loc="left")
    # HR panel
    ax = axes[1]
    ax.plot(tmin, p["hr"], color=MUTED, lw=0.8, label="recorded HR")
    st, ok = W["start"], W["ok"]
    tw = (st + 15) / 60.0
    b = (r.hr_max - r.hr_rest) / (a["vo2max"] * W["A"] - ph.VO2_REST)
    hr_fit = r.hr_rest + b * W["x0"] * (1 + a["drift"] * W["t"])
    ax.plot(tw[ok], hr_fit[ok], ".", color=C["ASCENT"], ms=3.5, label="ASCENT fitted HR (lagged demand)")
    ax.set_ylabel("HR (bpm)")
    ax.legend(loc="lower right", ncol=2)
    # per-window VO2max panel
    ax = axes[2]
    gok = GW["ok"]
    ax.plot((GW["start"][gok] + 15) / 60.0, GW["vmax"][gok], MARK["G_trail"], color=C["G_trail"],
            ms=3.2, mec="none", alpha=0.85, label="Garmin-like trail: per-window values")
    yv = (W["hr"] - r.hr_rest) / (1 + a["drift"] * W["t"])
    vw = (ph.VO2_REST + (r.hr_max - r.hr_rest) * W["x0"] / np.maximum(yv, 1e-6)) / W["A"]
    w = np.zeros(st.size)
    w[ok] = W["w_ok"]
    sz = 2 + 40 * w / w.max()
    ax.scatter(tw[ok], vw[ok], s=sz[ok], color=C["ASCENT"], alpha=0.7, lw=0,
               label="ASCENT: per-window values (size = weight)")
    ax.axhline(r.target, color=TRUTH, lw=1.0, label=f"truth {r.target:.1f}")
    ax.axhline(g["vo2max"], color=C["G_trail"], lw=1.2, ls="--",
               label=f"Garmin-like session value {g['vo2max']:.1f}")
    ax.axhline(a["vo2max"], color=C["ASCENT"], lw=1.2, ls="--",
               label=f"ASCENT session value {a['vo2max']:.1f}")
    ax.set_ylim(5, 75)
    ax.set_ylabel("VO$_2$max (ml·kg$^{-1}$·min$^{-1}$)")
    ax.set_xlabel("Time (min)")
    ax.legend(loc="lower center", ncol=2, fontsize=6.8)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_example_session.pdf"))
    return dict(target=r.target, garmin=g["vo2max"], ascent=a["vo2max"], tau=a["tau"],
                drift=a["drift"], shares=a["weight_share"])


MECH_LABEL = {"gps_bias": "GPS path shortening", "terrain": "terrain roughness (unknown η)",
              "ecc": "HR excess on descents", "kinetics": "HR kinetics (lag)",
              "drift": "cardiac drift", "altitude": "altitude", "stops": "stops",
              "walk": "power-hiking", "econ_mismatch": "individual economies",
              "curvature": "HR–VO$_2$ curvature", "obstacles": "obstacles"}


def fig_attribution(kinds=("trail_runnable", "mountain", "skyrun")):
    df = pd.read_pickle(os.path.join(RES, "attribution.pkl"))
    ests = ["G_trail", "G_trail_nodesc"]
    for e in ests:
        df["e_" + e] = df[e] - df["target"]
    base = df[df.cond == "all"].set_index(["seed", "kind"])
    mechs = [m for m in MECH_LABEL if ("no_" + m) in set(df.cond) and m != "altitude"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.6, 3.1), sharey=True)
    for ax, kind in zip(axes, kinds):
        y = np.arange(len(mechs))
        for j, e in enumerate(ests):
            vals = []
            for m in mechs:
                d = df[(df.cond == "no_" + m) & (df.kind == kind)].set_index(["seed", "kind"])
                b = base.loc[d.index, "e_" + e]
                vals.append(np.nanmean(b.values - d["e_" + e].values))
            ax.barh(y + (j - 0.5) * 0.38, vals, height=0.34, color=C[e], label=SHORT[e])
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([MECH_LABEL[m] for m in mechs])
        ax.invert_yaxis()
        ax.set_title(LABEL[kind], loc="left")
        ax.set_xlabel("Contribution to bias")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, os.path.join(FIG, "fig_attribution.pdf"))


def fig_ablation():
    out = {}
    for nm in ("ablation", "ablation_matched"):
        fp = os.path.join(RES, nm + ".pkl")
        if os.path.exists(fp):
            d = pd.read_pickle(fp)
            d["err"] = d["est"] - d["target"]
            out[nm] = d
    kinds = ["road_flat", "trail_runnable", "mountain", "skyrun"]
    variants = list(out["ablation"]["variant"].drop_duplicates())
    fig, axes = plt.subplots(1, len(out), figsize=(6.6, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    seq = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    for ax, (nm, d) in zip(axes, out.items()):
        y = np.arange(len(variants))
        for j, k in enumerate(kinds):
            rm = [np.sqrt(np.nanmean(d[(d.variant == v) & (d.kind == k)]["err"] ** 2)) for v in variants]
            ax.plot(rm, y + (j - 1.5) * 0.12, "o", color=seq[j], ms=4, label=LABEL[k])
        ax.set_yticks(y)
        ax.set_yticklabels(variants)
        ax.set_xlabel("RMSE (ml·kg$^{-1}$·min$^{-1}$)")
        ax.set_title("Full simulated world" if nm == "ablation" else "World without unmodelled terrain/GPS cost",
                     loc="left", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, os.path.join(FIG, "fig_ablation.pdf"))


SENS_LABEL = {"ecc_lambda": "HR excess on descents λ", "gps_scale": "GPS shortening (× nominal)",
              "terrain_scale": "terrain roughness (× nominal)", "tau_slow": "slow HR time constant (s)",
              "drift": "cardiac drift (per h)", "walk_econ": "hiking economy (× population)",
              "alt_slope": "VO$_2$max loss per 1000 m", "hrmax_err": "HRmax setting error (bpm)",
              "hrrest_err": "HRrest setting error (bpm)"}


def fig_sensitivity():
    df = pd.read_pickle(os.path.join(RES, "sensitivity.pkl"))
    ests = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
    d = df[(df.kind == "mountain")]
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 5.6), sharey=True)
    for ax, f in zip(axes.ravel(), SENS_LABEL):
        x = d[d.factor == f]
        for e in ests:
            g = (x[e] - x["target"]).groupby(x["level"])
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values / np.sqrt(g.count().values) * 1.96,
                        color=C[e], marker=MARK[e], ms=3.5, lw=1.2, elinewidth=0.8, capsize=0,
                        label=SHORT[e])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(SENS_LABEL[f], loc="left", fontsize=7.8)
        ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Bias (ml·kg$^{-1}$·min$^{-1}$)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7.0, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, os.path.join(FIG, "fig_sensitivity.pdf"))


def _real_markers(d):
    mk = np.where(d["race"], "*", np.where(d["surface"] == "road", "o", "^"))
    return mk


def fig_real_scatter():
    df = pd.read_csv(os.path.join(RES, "real_estimates_hrmax170.csv"), dtype={"id": str})
    ests = ["G_flat", "G_trail", "ASCENT"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
    for ax, e in zip(axes, ests):
        for kind, mk, lab in (("road", "o", "road (training)"), ("trail", "^", "trail (training)"),
                              ("race", "*", "trail race / road race")):
            if kind == "race":
                d = df[df.race]
            else:
                d = df[(df.surface == kind) & ~df.race]
            ax.plot(d["dplus_per_km"], d[e], mk, color=C[e], ms=7 if mk == "*" else 4.2,
                    mec=C[e] if kind != "road" else "white", mew=0.9 if kind == "trail" else 0.5,
                    alpha=0.9, label=lab, mfc=C[e] if kind != "trail" else "white")
        tr = df[~df.race].dropna(subset=[e])
        b, a0 = np.polyfit(tr["dplus_per_km"], tr[e], 1)
        xs = np.linspace(0, 90, 10)
        ax.plot(xs, a0 + b * xs, color=INK2, lw=1.0)
        ax.text(0.98, 0.04, f"training slope {10 * b:+.1f}\nper 10 m/km", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_title(SHORT[e], loc="left")
        ax.set_xlabel("Climbing density D+ (m/km)")
    axes[0].set_ylabel("Session VO$_2$max estimate")
    axes[0].legend(loc="upper right", fontsize=6.5, handletextpad=0.2)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_real_scatter.pdf"))


def fig_real_time():
    df = pd.read_csv(os.path.join(RES, "real_estimates_hrmax170.csv"), dtype={"id": str})
    df["dt"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for e in ("G_trail", "ASCENT"):
        for kind, mk in (("road", "o"), ("trail", "^")):
            d = df[(df.surface == kind) & ~df.race]
            ax.plot(d["dt"], d[e], mk, color=C[e], ms=3.8, alpha=0.75,
                    mfc=C[e] if kind == "road" else "white")
        d = df[df.race]
        ax.plot(d["dt"], d[e], "*", color=C[e], ms=7)
    ax.step(df["dt"], df["disp_G_trail"], where="post", color=C["G_trail"], lw=1.6,
            label="Garmin-like trail, constant-gain display")
    ax.step(df["dt"], df["disp_ASCENT_race"], where="post", color=C["ASCENT"], lw=1.6,
            label="ASCENT, race-aware Kalman display")
    ax.plot([], [], "o", color=MUTED, ms=3.8, label="road training")
    ax.plot([], [], "^", color=MUTED, mfc="white", ms=3.8, label="trail training")
    ax.plot([], [], "*", color=MUTED, ms=7, label="race")
    ax.set_ylabel("VO$_2$max (ml·kg$^{-1}$·min$^{-1}$)")
    ax.legend(loc="lower left", ncol=3, fontsize=6.8)
    ax.set_ylim(30, 75)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_real_time.pdf"))


def fig_real_sessions(ids=("20374217194", "19095176093")):
    wins = pd.read_pickle(os.path.join(RES, "real_windows_hrmax170.pkl"))
    df = pd.read_csv(os.path.join(RES, "real_estimates_hrmax170.csv"), dtype={"id": str}).set_index("id")
    fig, axes = plt.subplots(3, len(ids), figsize=(6.6, 5.0), sharex="col",
                             gridspec_kw=dict(height_ratios=[0.9, 1.1, 1.4]))
    for j, aid in enumerate(ids):
        p, W, GW = wins[(aid, "prep")], wins[(aid, "ASCENT")], wins[(aid, "G_trail")]
        info = df.loc[aid]
        tmin = p["t"] / 60.0
        ax = axes[0, j]
        ax.plot(tmin, p["zs"], color=INK2, lw=1.1)
        ax.set_title(f"{info['date']}  {'race: ' + info['race_name'] if info['race'] else info['name'][:28]}",
                     loc="left", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("Altitude (m)")
        ax = axes[1, j]
        ax.plot(tmin, p["hr"], color=MUTED, lw=0.7)
        st, ok = W["start"], W["ok"]
        tw = (st + 15) / 60.0
        V, dr = info["ASCENT"], info["ASCENT_drift"]
        b = (170.0 - 52.0) / (V * W["A"] - ph.VO2_REST)
        ax.plot(tw[ok], 52.0 + b[ok] * W["x0"][ok] * (1 + dr * W["t"][ok]), ".", color=C["ASCENT"], ms=3)
        if j == 0:
            ax.set_ylabel("HR (bpm)")
        ax = axes[2, j]
        gok = GW["ok"]
        ax.plot((GW["start"][gok] + 15) / 60.0, GW["vmax"][gok], "s", color=C["G_trail"], ms=2.8,
                mec="none", alpha=0.85, label="Garmin-like trail, per window")
        yv = (W["hr"] - 52.0) / (1 + dr * W["t"])
        vw = (ph.VO2_REST + 118.0 * W["x0"] / np.maximum(yv, 1e-6)) / W["A"]
        w = np.zeros(st.size)
        w[ok] = W["w_ok"]
        ax.scatter(tw[ok], vw[ok], s=2 + 30 * w[ok] / w.max(), color=C["ASCENT"], alpha=0.7, lw=0,
                   label="ASCENT, per window (size = weight)")
        ax.axhline(info["G_trail"], color=C["G_trail"], ls="--", lw=1.1)
        ax.axhline(V, color=C["ASCENT"], ls="--", lw=1.1)
        ax.set_ylim(20, 90)
        ax.set_xlabel("Time (min)")
        if j == 0:
            ax.set_ylabel("VO$_2$max (ml·kg$^{-1}$·min$^{-1}$)")
            ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_real_sessions.pdf"))


def longitudinal_tables():
    df = pd.read_pickle(os.path.join(RES, "longitudinal.pkl"))
    rows, traj = [], {}
    for (sc, seed), d in df.sort_values("day").groupby(["scenario", "seed"]):
        trk = {"Garmin-like trail + constant gain": es.EWMADisplay(0.3),
               "Garmin-like no-descent + constant gain": es.EWMADisplay(0.3),
               "ASCENT + Kalman": es.KalmanTracker(),
               "ASCENT + race-aware Kalman": es.RaceAwareTracker()}
        prev = {k: None for k in trk}
        for _, r in d.iterrows():
            for k, t in trk.items():
                if k.startswith("Garmin-like no-descent"):
                    m = t.update(r["day"], r["G_trail_nodesc"])
                elif k.startswith("Garmin"):
                    m = t.update(r["day"], r["G_trail"])
                elif "race-aware" in k:
                    m = t.update(r["day"], r["ASCENT"], r["ASCENT_sd"], race=bool(r["race"]))
                else:
                    m = t.update(r["day"], r["ASCENT"], r["ASCENT_sd"])
                cat = ("race" if r["race"] else {"road_flat": "road", "road_hilly": "road",
                       "trail_runnable": "runnable trail", "mountain": "mountain trail"}[r["kind"]])
                rows.append(dict(scenario=sc, seed=seed, day=r["day"], week=r["week"], tracker=k,
                                 cat=cat, display=m, target=r["target"],
                                 change=(np.nan if prev[k] is None or m is None else
                                         np.round(m) - np.round(prev[k]))))
                prev[k] = m
    out = pd.DataFrame(rows)
    out["err"] = out["display"] - out["target"]
    return out


def fig_longitudinal():
    fp = os.path.join(RES, "longitudinal_tracks.pkl")
    if os.path.exists(fp):
        out = pd.read_pickle(fp)
    else:
        out = longitudinal_tables()
        out.to_pickle(fp)                       # make_tables.tab_long reads this file
    trackers = ["Garmin-like trail + constant gain", "Garmin-like no-descent + constant gain",
                "ASCENT + Kalman", "ASCENT + race-aware Kalman"]
    col = {trackers[0]: C["G_trail"], trackers[1]: C["G_trail_nodesc"], trackers[2]: "#86b6ef",
           trackers[3]: C["ASCENT"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    d = out[(out.scenario == "race_effect")]
    seed = sorted(d.seed.unique())[3]
    e = d[d.seed == seed]
    t0 = e[e.tracker == trackers[0]]
    ax.plot(t0["day"] / 7, t0["target"], color=TRUTH, lw=1.0, label="true VO$_2$max")
    for k in trackers:
        x = e[e.tracker == k]
        ax.step(x["day"] / 7, np.round(x["display"]), where="post", color=col[k], lw=1.5,
                label=k + " (rounded display)")
    rc = t0[t0.cat == "race"]
    ylo = np.nanmin(np.round(e["display"])) - 2.5
    ax.plot(rc["day"] / 7, np.full(len(rc), ylo + 0.6), "*", color=INK2, ms=7, label="race day")
    ax.set_ylim(ylo, np.nanmax(t0["target"]) + 1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Displayed VO$_2$max")
    ax.set_title("One simulated runner (race-day HR inflation on)", loc="left", fontsize=8)
    ax = axes[1]
    cats = ["road", "runnable trail", "mountain trail", "race"]
    y = np.arange(len(cats))
    for j, k in enumerate(trackers):
        m = [d[(d.tracker == k) & (d.cat == c)]["change"].mean() for c in cats]
        ax.barh(y + (j - 1.5) * 0.2, m, height=0.18, color=col[k])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel("Mean change of the displayed value")
    ax.grid(axis="y", visible=False)
    ax.set_title("After each session type, 40 runners", loc="left", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.8, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, os.path.join(FIG, "fig_longitudinal.pdf"))
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["main", "example"]
    if "main" in what:
        fig_main()
    if "example" in what:
        print(fig_example())
    if "attribution" in what:
        fig_attribution()
    if "ablation" in what:
        fig_ablation()
    if "sensitivity" in what:
        fig_sensitivity()
    if "longitudinal" in what:
        o = fig_longitudinal()
        print(o.groupby(["scenario", "tracker"]).apply(lambda x: np.sqrt(np.nanmean(x[x.week >= 2]["err"] ** 2))).round(2))
        print(o.groupby(["scenario", "tracker", "cat"])["change"].mean().round(2).unstack())
    if "real" in what:
        fig_real_scatter()
        fig_real_time()
        fig_real_sessions()
