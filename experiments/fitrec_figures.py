"""Figure for the FitRec terrain-invariance test: dose-response by climb bin and within-runner slopes."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import C, INK2, MARK, SHORT, plt, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
LABELS = ["8-15", "15-25", "25-40", ">40"]


def fig_fitrec():
    t = pd.read_csv(os.path.join(RES, "fitrec_dose_summary.csv"))
    fx = pd.read_csv(os.path.join(RES, "fitrec_effects.csv")).set_index("estimator")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    x = np.arange(len(LABELS))
    for j, e in enumerate(EST):
        d = t[t.estimator == e].set_index("bin").loc[LABELS]
        off = (j - 2) * 0.06
        ax.plot(x + off, d["trim"], "-", color=C[e], lw=1.3, zorder=2)
        ax.plot(x + off, d["trim"], MARK[e], color=C[e], ms=4.5, mec="white", mew=0.6, zorder=3, label=SHORT[e])
    d = t[t.estimator == "ASCENT_gh"].set_index("bin").loc[LABELS]
    ax.plot(x + 0.12, d["trim"], "--", color=C["ASCENT"], lw=1.0, zorder=2, alpha=0.7,
            label="ASCENT, speed-based walking heuristic")
    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Climbing of the run (m per km)")
    ax.set_ylabel("Estimate minus the runner's flat runs\n(ml·kg$^{-1}$·min$^{-1}$, trimmed mean over runners)")
    nr = pd.read_pickle(os.path.join(RES, "fitrec_selected.pkl")).userId.nunique()
    ax.set_title(f"Dose-response, {nr} runners", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.3, loc="lower left", ncol=1, handlelength=1.8)
    ax = axes[1]
    for j, e in enumerate(EST):
        y = j
        ax.plot([fx.loc[e, "slope_lo"], fx.loc[e, "slope_hi"]], [y, y], color=C[e], lw=1.2, zorder=2)
        ax.plot(fx.loc[e, "slope10"], y, MARK[e], color=C[e], ms=5, mec="white", mew=0.6, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(range(len(EST)))
    ax.set_yticklabels([SHORT[e] for e in EST])
    ax.invert_yaxis()
    ax.set_xlabel("Within-runner slope per 10 m/km, duration held fixed\n(mean and 95 % CI over runners)")
    ax.set_title(f"Slope on climbing, {int(fx.loc['ASCENT', 'n_users_slope'])} runners", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_fitrec.pdf"))


if __name__ == "__main__":
    fig_fitrec()
