"""Shared matplotlib style (validated categorical palette, recessive chrome)."""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots (validated: scripts/validate_palette.js, light mode)
C = {"ASCENT": "#2a78d6", "G_trail": "#eb6834", "G_flat": "#1baf7a", "G_trail_hrgap": "#eda100",
     "G_trail_nodesc": "#e87ba4"}
MARK = {"ASCENT": "o", "G_trail": "s", "G_flat": "^", "G_trail_hrgap": "D", "G_trail_nodesc": "v"}
NAME = {"ASCENT": "ASCENT (this work)", "G_trail": "Garmin-like, trail (GAP + terrain + altitude)",
        "G_flat": "Garmin-like, classic (level running)",
        "G_trail_hrgap": "Garmin-like, trail with HR-calibrated GAP",
        "G_trail_nodesc": "Garmin-like, trail without descents"}
SHORT = {"ASCENT": "ASCENT", "G_trail": "Garmin-like trail", "G_flat": "Garmin-like classic",
         "G_trail_hrgap": "Garmin-like HR-GAP", "G_trail_nodesc": "Garmin-like no-descent"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRUTH = "#0b0b0b"

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
