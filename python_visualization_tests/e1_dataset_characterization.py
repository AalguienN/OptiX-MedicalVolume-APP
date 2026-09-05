# ============================================================
# E1 — Dataset characterization (4 plots).
#   1. Distribution : on-disk footprint (MB, linear) per series.
#   2. Distribution : sparsity (%) per series.
#   3. Scatter      : voxel resolution (X*Y in-plane vs Z depth),
#      colour = sparsity, size = total voxels, CT/MR markers.
#   4. Image grid   : the 47 manual-mode snapshots, each labelled
#      with its disk footprint (MB) and sparsity at eps = 0.01.
#   Usage: venv/bin/python e1_dataset_characterization.py
# ============================================================
import os
import ast
import glob

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

import viz_common as vc

RES_ROOT = os.path.dirname(vc.CSV_PATH)


def metric_frame():
    df = vc.load()
    df = df[df["mode"] == "manual"].sort_values("dataset").drop_duplicates("dataset")

    def parse_tuple(s):
        return ast.literal_eval(s) if isinstance(s, str) else tuple(s)

    dims = df["vol_dims"].map(lambda s: tuple(int(v) for v in s.split("x")))
    sp = df["spacing"].map(parse_tuple)

    disk = {}
    for f in glob.glob(os.path.join(RES_ROOT, "*/dataset_info.csv")):
        with open(f) as fh:
            for row in pd.read_csv(fh).itertuples():
                disk[row.dataset] = row.disk_bytes

    out = pd.DataFrame({
        "dataset": df["dataset"],
        "modality": df["modality"],
        "slices": df["num_slices"],
        "vol_dims": df["vol_dims"],
        "spacing": [tuple(round(v, 3) for v in t) for t in sp],
        "disk_mb": df["dataset"].map(disk).div(1e6),
        "inplane": df["slice_dims"].map(lambda s: int(s.split("x")[0]) * int(s.split("x")[1])),
        "voxels_m": df["n_total"] / 1e6,
        "total_voxels": df["n_total"],
        "sparsity_pct": df["sparsity_pct"],
        "snapshot_path": df["snapshot_path"],
    })
    return out


def dist_plot(d, col, xlabel, title, name, bins):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(d, x=col, hue="modality", multiple="stack", bins=bins, ax=ax,
                 palette={"CT": "#4C72B0", "MR": "#DD8452"},
                 edgecolor="white", linewidth=0.3)
    ax.set_xticks(bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    # ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    s = d[col]
    med, mean, lo, hi = s.median(), s.mean(), s.min(), s.max()
    ax.text(0.98, 0.96,
            f"n = {len(s)}\nmin = {lo:,.1f}\nmedian = {med:,.1f}\n"
            f"mean = {mean:,.1f}\nmax = {hi:,.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    fig.tight_layout()
    vc.save(fig, name)
    plt.close(fig)


def image_grid(d):
    n = len(d)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    first = np.asarray(Image.open(d["snapshot_path"].iloc[0]).convert("RGB"))
    img_h, img_w = first.shape[:2]
    cell_w = 3.4
    cell_h = cell_w * img_h / img_w + 0.1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, cell_h * nrows + 0.6))
    fig.subplots_adjust(hspace=0.000, wspace=0.02, left=0.005, right=0.995,
                        bottom=0.005, top=0.97)
    axes = np.atleast_2d(axes)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            i = r * ncols + c
            ax.axis("off")
            if i >= n:
                continue
            row = d.iloc[i]
            img = np.asarray(Image.open(row["snapshot_path"]).convert("RGB"))
            ax.imshow(img[::-1], interpolation="nearest")
            mod, modc = ("CT", "#4C72B0") if row["modality"] == "CT" else ("MR", "#E5AE38")
            badges = [
                (f"{row['disk_mb']:.0f} MB", "#8E44AD", 0.015, "left"),
                (f"{row['sparsity_pct']:.1f}%", "#D62728", 0.185, "left"),
                (mod, modc, 0.985, "right"),
            ]
            for txt, color, x, ha in badges:
                ax.text(x, 0.97, txt, transform=ax.transAxes, ha=ha, va="top",
                        fontsize=8, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none",
                                  alpha=0.9))
    # fig.suptitle("Manual-mode snapshots of all 47 series (disk MB, sparsity %)",
    #              fontsize=25, y=0.995)
    vc.save(fig, "e1_manual_snapshots.png")
    plt.close(fig)


def scatter_voxels_plot(d):
    dropped = d[d["inplane"] != 512 * 512]
    d = d[d["inplane"] == 512 * 512]
    print(f"  scatter: keeping {len(d)} series, dropped non-512x512 in-plane outliers:")
    for _, r in dropped.iterrows():
        print(f"    - {r['dataset']} ({r['vol_dims']})")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(d["sparsity_pct"].min(),
                                                  d["sparsity_pct"].max()))
    sm.set_array([])
    for mod, m in [("CT", "o"), ("MR", "^")]:
        sel = d[d["modality"] == mod]
        ax.scatter(sel["voxels_m"], sel["slices"].astype(float),
                   c=sel["sparsity_pct"], s=sel["disk_mb"] * 0.8, cmap="viridis",
                   vmin=d["sparsity_pct"].min(), vmax=d["sparsity_pct"].max(),
                   marker=m, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("total voxels (x\u00d7y\u00d7z, millions, log)")
    ax.set_ylabel("depth (Z slices)")
    ax.set_title("Voxel resolution per series (size = disk MB, 3 non-512\u00b2 MR removed)")
    ax.grid(alpha=0.3)
    fig.colorbar(sm, ax=ax, label="sparsity (%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color="gray", label="CT"),
               Line2D([0], [0], marker="^", ls="", color="gray", label="MR")]
    ax.legend(handles=handles, loc="upper left", title="modality")
    fig.tight_layout()
    vc.save(fig, "e1_scatter_voxels.png")
    plt.close(fig)


def main():
    d = metric_frame()
    print(f"{len(d)} series loaded")

    dist_plot(d, "disk_mb", "on-disk size (MB)",
              "Distribution of disk footprint per series",
              "e1_dist_disk_mb.png", bins=list(range(0, 401, 50)))
    dist_plot(d, "sparsity_pct", "sparsity (%)",
              "Distribution of sparsity per series (eps = 0.01)",
              "e1_dist_sparsity.png", bins=list(range(40, 101, 10)))
    scatter_voxels_plot(d)
    image_grid(d)


if __name__ == "__main__":
    main()
