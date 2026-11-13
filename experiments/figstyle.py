"""Shared matplotlib style (validated categorical palette, recessive chrome)."""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots (first revision): a colour-blind-safe set (Okabe & Ito) in which no two series
# share a hue family, and every series also has its own marker, line style and hatch, so that the
# figures can be read without colour.
C = {"ASCENT": "#000000", "G_trail": "#009E73", "G_flat": "#D55E00", "G_trail_hrgap": "#56B4E9",
     "G_trail_nodesc": "#CC79A7"}
MARK = {"ASCENT": "o", "G_trail": "s", "G_flat": "^", "G_trail_hrgap": "D", "G_trail_nodesc": "v"}
LS = {"ASCENT": "-", "G_trail": "--", "G_flat": "-", "G_trail_hrgap": "-.", "G_trail_nodesc": ":"}
HATCH = {"ASCENT": "", "G_trail": "////", "G_flat": "", "G_trail_hrgap": "xxxx", "G_trail_nodesc": "...."}
NAME = {"ASCENT": "ASCENT (proof of concept)", "G_trail": "Reconstruction: trail (metabolic grade adjustment)",
        "G_flat": "Reconstruction: classic (level running)",
        "G_trail_hrgap": "Reconstruction: HR-GAP (HR-calibrated grade adjustment)",
        "G_trail_nodesc": "Reconstruction: no-descent"}
SHORT = {"ASCENT": "ASCENT", "G_trail": "Trail (reconstruction)", "G_flat": "Classic (reconstruction)",
         "G_trail_hrgap": "HR-GAP (reconstruction)", "G_trail_nodesc": "No-descent (reconstruction)"}
UNIT = "ml·kg$^{-1}$·min$^{-1}$"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRUTH = "#8a8a8a"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def save(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
