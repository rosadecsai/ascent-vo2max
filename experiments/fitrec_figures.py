"""Figure of the field test (first revision): terrain-associated difference by climbing class with
bootstrap intervals, and within-runner slopes.  Reads results/revision/gaps.csv (written by
fitrec_revision_analysis.py), results/fitrec_effects.csv and results/fitrec_noise.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import C, INK2, LS, MARK, UNIT, plt, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
EST = ["G_flat", "G_trail", "G_trail_nodesc", "G_trail_hrgap", "ASCENT"]
LAB = {"G_flat": "Classic (recon.)", "G_trail": "Trail (recon.)", "G_trail_nodesc": "No-descent (recon.)",
       "G_trail_hrgap": "HR-GAP (recon.)", "ASCENT": "ASCENT v1, no gait labels"}
LABELS = ["8-15", "15-25", "25-40", ">40"]
OPEN = ("G_trail", "G_trail_nodesc")


def fig_fitrec():
    t = pd.read_csv(os.path.join(RES, "revision", "gaps.csv"))
    fx = pd.read_csv(os.path.join(RES, "fitrec_effects.csv")).set_index("estimator")
    nz = pd.read_csv(os.path.join(RES, "fitrec_noise.csv"), index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]
    x = np.arange(len(LABELS))
    sd0 = float(nz.loc["G_flat", "control_sd"])
    ax.axhspan(-sd0, sd0, color="#d9d9d9", alpha=0.55, lw=0, zorder=0, label="flat-versus-flat control, ±1 SD")
    for j, e in enumerate(EST):
        d = t[t.label == f"all|{e}"].set_index("bin").loc[LABELS]
        off = (j - 2) * 0.13
        ax.plot(x + off, d["trim"], LS[e], color=C[e], lw=1.1, zorder=2)
        ax.errorbar(x + off, d["trim"], yerr=[d["trim"] - d["lo"], d["hi"] - d["trim"]], fmt=MARK[e], color=C[e],
                    ms=4.8, mfc=("white" if e in OPEN else C[e]), mec=C[e], mew=1.0, elinewidth=1.0, capsize=1.8,
                    zorder=3, label=LAB[e])
    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["8–15", "15–25", "25–40", ">40"])
    ax.set_xlabel("Climbing of the run (m of elevation gain per km)")
    ax.set_ylabel("Estimate minus the same runner's\nflat runs (" + UNIT + ")")
    ax.set_title("Terrain-associated difference", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.2, loc="lower left", ncol=1, handlelength=2.2)
    ax = axes[1]
    for j, e in enumerate(EST):
        ax.errorbar(fx.loc[e, "slope10"], j, xerr=[[fx.loc[e, "slope10"] - fx.loc[e, "slope_lo"]],
                                                    [fx.loc[e, "slope_hi"] - fx.loc[e, "slope10"]]],
                    fmt=MARK[e], color=C[e], ms=5.2, mfc=("white" if e in OPEN else C[e]), mec=C[e], mew=1.0,
                    elinewidth=1.1, capsize=2, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(range(len(EST)))
    ax.set_yticklabels([LAB[e] for e in EST])
    ax.invert_yaxis()
    ax.set_xlabel("Change of the estimate per 10 m/km\nof climbing (" + UNIT + ")")
    ax.set_title("Within-runner slope", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_r1_fitrec.pdf"))


if __name__ == "__main__":
    fig_fitrec()
