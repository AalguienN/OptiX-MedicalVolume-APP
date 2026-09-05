# ============================================================
# D8 — Image table (epsilon-sweep subset): only the strategy
#   traces that have a FULL epsilon sweep, 0.01 -> 0.50.
#   Rows: adaptive, brick16, octree-leaf8, octree-regions-leaf8,
#         nanovdb-nearest   (5 strategies x 5 epsilons grid)
#   Usage: venv/bin/python d8_image_grid_sweep.py [dataset]
# ============================================================
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import viz_common as vc

df = vc.load()
epsilons = sorted(df["epsilon"].unique())
EPS_CODE = {eps: f"{int(round(eps * 100)):03d}" for eps in epsilons}

# only the slots that span every epsilon
ROWS = [
    ("adaptive",        "eps"),
    ("bricked-regions", "brick16"),
    ("octree",          "leaf8"),
    ("octree-regions",  "leaf8"),
    ("nanovdb",         "nearest"),
]


def strat_for(mode, slot, eps):
    if slot == "eps":
        return f"{mode}_eps{EPS_CODE[eps]}"
    return f"{mode}_{slot}_eps{EPS_CODE[eps]}"


def _gray(a):
    """Grayscale luminance of an RGB uint8 array."""
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def diff_pcts(a, b):
    """% mean-abs-difference vs reference b, for luminance and per-channel RGB."""
    lum = 100.0 * np.mean(np.abs(_gray(a) - _gray(b))) / 255.0
    rgb = 100.0 * np.mean(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=2)) / 255.0
    return lum, rgb


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else df["dataset"].iloc[0]
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        sys.exit(f"Dataset '{dataset}' not found in the CSV.")

    cells = {eps: {} for eps in epsilons}
    for i, (mode, slot) in enumerate(ROWS):
        for eps in epsilons:
            name = strat_for(mode, slot, eps)
            hit = sub[(sub["mode"] == mode) & (sub["strategy"] == name)]
            if not hit.empty:
                cells[eps][i] = hit["snapshot_path"].iloc[0]

    man_rows = sub[sub["mode"] == "manual"]
    if man_rows.empty:
        sys.exit(f"No manual reference snapshot for dataset '{dataset}'.")
    man = np.asarray(Image.open(man_rows["snapshot_path"].iloc[0]), dtype=np.uint8)

    n_rows, n_cols = len(ROWS), len(epsilons)
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3.6 * (n_cols + 1), 1.9 * n_rows + 1.2))

    for i, (mode, slot) in enumerate(ROWS):
        axes[i, 0].text(0.5, 0.5, f"{mode} | {slot}", ha="center", va="center",
                        fontsize=9, wrap=True)
        axes[i, 0].axis("off")
        for j, eps in enumerate(epsilons):
            ax = axes[i, j + 1]
            if i in cells[eps]:
                img = np.asarray(Image.open(cells[eps][i]), dtype=np.uint8)
                ax.imshow(img, interpolation="nearest")
                lum_pct, rgb_pct = diff_pcts(img, man)
                ax.text(0.02, 0.98, f"\u0394 lum {lum_pct:.1f}%\n\u0394 rgb {rgb_pct:.1f}%",
                        transform=ax.transAxes, ha="left", va="top", fontsize=7,
                        color="white", linespacing=1.3,
                        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.45))
            ax.axis("off")
            if i == 0:
                ax.set_title(f"eps = {eps}", fontsize=9)

    axes[0, 0].set_title("strategy", fontsize=9)
    fig.suptitle(f"Epsilon-sweep snapshots per strategy — {dataset}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(vc.PLOT_DIR, f"d8_image_grid_sweep_{dataset}.png")
    fig.savefig(out, dpi=150)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()