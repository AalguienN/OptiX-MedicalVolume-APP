# ============================================================
# D7 — Image table: rows = strategies, columns = epsilon.
#   Given a dataset (series), renders the actual snapshot PPMs
#   in a grid so you can eyeball how each strategy degrades as
#   the error tolerance is relaxed.
#
#   Usage:
#     venv/bin/python d7_image_grid.py                       # first dataset
#     venv/bin/python d7_image_grid.py manifest-1787673532591_DICOM__series_1
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

# row = (mode, "slot"): slot None/plain -> only at eps 0.01; else eps-carrying
ROWS = [
    ("manual",                    None),
    ("optix",                     None),
    ("adaptive",                  "eps"),
    ("bricked-regions",           "brick8"),
    ("bricked-regions",           "brick32"),
    ("bricked-regions",           "brick16"),
    ("octree",                    "leaf4"),
    ("octree",                    "leaf8_adapt"),
    ("octree",                    "leaf8"),
    ("octree-regions",            "leaf4"),
    ("octree-regions",            "leaf8_adapt"),
    ("octree-regions",            "leaf8"),
    ("nanovdb",                   "trilinear"),
    ("nanovdb",                   "nearest"),
]


EPS_SLOTS = {"eps", "brick16", "leaf8", "nearest"}   # slots that carry an epsilon suffix


def strat_for(mode, slot, eps):
    """Strategy name for a (mode, slot) at a given epsilon (None = no such file)."""
    if slot is None:                        # manual / optix
        return mode if eps == 0.01 else None
    if slot in EPS_SLOTS:
        if slot == "eps":
            return f"{mode}_eps{EPS_CODE[eps]}"
        return f"{mode}_{slot}_eps{EPS_CODE[eps]}"
    return f"{mode}_{slot}" if eps == 0.01 else None   # plain strategies: eps 0.01 only


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else df["dataset"].iloc[0]
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        sys.exit(f"Dataset '{dataset}' not found in the CSV.")

    cells = {eps: {} for eps in epsilons}          # eps -> {row_idx: snapshot_path}
    used = {}
    for i, (mode, slot) in enumerate(ROWS):
        for eps in epsilons:
            name = strat_for(mode, slot, eps)
            if name is None:
                continue
            hit = sub[(sub["mode"] == mode) & (sub["strategy"] == name) & (sub["epsilon"] == eps)]
            if not hit.empty:
                cells[eps][i] = hit["snapshot_path"].iloc[0]
                used[(i, eps)] = name

    n_rows, n_cols = len(ROWS), len(epsilons)
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3.6 * (n_cols + 1), 1.9 * n_rows + 1.2))

    for i, (mode, slot) in enumerate(ROWS):
        label = mode if slot is None else f"{mode} | {slot}"
        axes[i, 0].text(0.5, 0.5, label, ha="center", va="center",
                        fontsize=8, wrap=True)
        axes[i, 0].axis("off")
        for j, eps in enumerate(epsilons):
            ax = axes[i, j + 1]
            if i in cells[eps]:
                img = np.asarray(Image.open(cells[eps][i]), dtype=np.uint8)
                ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"eps = {eps}", fontsize=9)

    axes[0, 0].set_title("strategy", fontsize=9)
    fig.suptitle(f"Rendered snapshots per strategy x epsilon — {dataset}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(vc.PLOT_DIR, f"d7_image_grid_{dataset}.png")
    fig.savefig(out, dpi=150)
    print(f"  saved {out}")

    print("=== strategy used per cell (row / epsilon) ===")
    for (i, eps), name in sorted(used.items()):
        mode = ROWS[i][0]
        print(f"  {mode+(' | '+ROWS[i][1] if ROWS[i][1] else '') :34s} eps={eps:<5} {name}")


if __name__ == "__main__":
    main()