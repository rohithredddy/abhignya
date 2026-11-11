"""Matplotlib style for IEEE print: small fonts, grayscale-readable markers/hatches.

Implements CONTRIBUTING.md §7 (consistent style, readable in grayscale).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE_COL = 3.5   # inches, IEEE single column
DOUBLE_COL = 7.16  # inches, IEEE double column
GRAYS = ["#000000", "#555555", "#888888", "#BBBBBB", "#DDDDDD"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "x"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "++"]
LAB_ORDER = ["HGB", "K", "CREAT", "PLT"]


def apply_style() -> None:
    """Set global rcParams for all project figures."""
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
        "axes.prop_cycle": matplotlib.cycler(color=GRAYS[:4]),
        "figure.dpi": 100, "pdf.fonttype": 42,
    })


def figure(width: float = SINGLE_COL, height: float = 2.4, ncols: int = 1, nrows: int = 1,
           **kw):
    """Create a styled figure; returns (fig, axes)."""
    apply_style()
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)
