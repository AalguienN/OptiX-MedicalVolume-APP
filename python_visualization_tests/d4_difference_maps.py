# ============================================================
# D4 — Difference maps: for one representative dataset, show
#   the manual reference, a strategy render, and the |Δ| as a
#   heatmap at eps 0.01 / 0.10 / 0.50. Where does the error
#   appear — and how much do the families differ?
#
#   Files are taken straight from the snapshot_path column of
#   the results CSV (no filename guessing).
# ============================================================
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import viz_common as vc

RAW = "/home/adri/projects/TFM/optix_test_results/results.csv"
df = vc.load()

fams = ["adaptive", "octree", "bricked-regions"]
epsilons = [0.01, 0.10, 0.50]
sample_dataset = "manifest-1787673532591_DICOM__series_1"
sub = df[df["dataset"] == sample_dataset]


def gray(path):
    a = np.asarray(Image.open(path), dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def pick_snapshot(fam, eps):
    """snapshot_path of a deterministic eps-carrying variant of (fam, eps)."""
    g = sub[(sub["mode"] == fam) & (sub["epsilon"] == eps)]
    code = f"eps{int(round(eps * 1000)):03d}"
    hit = g[g["strategy"].astype(str).str.endswith(code)]
    if hit.empty:
        hit = g
    return hit["snapshot_path"].iloc[0]


man = gray(sub[(sub["mode"] == "manual")]["snapshot_path"].iloc[0])

fig, axes = plt.subplots(len(fams), len(epsilons) + 1, figsize=(22, 12))
for r, fam in enumerate(fams):
    for c, eps in enumerate(epsilons):
        path = pick_snapshot(fam, eps)
        img = gray(path)
        diff = np.abs(img - man)
        axes[r, c].imshow(diff, cmap="hot", vmin=0, vmax=150)
        axes[r, c].set_title(os.path.basename(path).replace(".ppm", ""), fontsize=8)
        axes[r, c].axis("off")
    axes[r, 0].imshow(man, cmap="gray")
    axes[r, 0].set_title("manual (reference)", fontsize=9)
    axes[r, 0].axis("off")

for r, fam in enumerate(fams):
    axes[r, 0].set_ylabel(fam, rotation=0, fontsize=11, labelpad=30)
fig.suptitle(f"Absolute luminance difference vs manual — {sample_dataset} "
             "(hot = bigger error)", fontsize=13)
fig.tight_layout()
vc.save(fig, "d4_difference_maps.png")