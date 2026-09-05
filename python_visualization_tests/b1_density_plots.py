# ============================================================
# B1 — Density plots: 1-D KDE (log render time, FPS) and 2-D
#   KDE contours (FPS vs GPU memory, FPS vs sparsity), split
#   by strategy family. Low = density.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

sns.kdeplot(data=df, x="render_mean_ms", hue="mode", common_norm=False,
            fill=True, alpha=0.25, ax=axes[0, 0])
axes[0, 0].set_xscale("log")
axes[0, 0].set_title("Density of render time (log-x)")

sns.kdeplot(data=df, x="fps_mean", hue="mode", common_norm=False,
            fill=True, alpha=0.25, ax=axes[0, 1])
axes[0, 1].set_xscale("log")
axes[0, 1].set_title("Density of FPS (log-x)")

sns.kdeplot(data=df[df["mode"].isin(["manual", "optix", "adaptive"])],
            x="gpu_mem_gb", y="fps_mean", hue="mode", thresh=0.05,
            fill=True, alpha=0.3, ax=axes[1, 0])
axes[1, 0].set_title("2-D density: FPS vs GPU mem (3 reference families)")

sns.kdeplot(data=df[df["mode"].isin(["manual", "nanovdb", "octree"])],
            x="sparsity_pct", y="fps_mean", hue="mode", thresh=0.05,
            fill=True, alpha=0.3, ax=axes[1, 1])
axes[1, 1].set_title("2-D density: FPS vs sparsity (3 families)")

fig.tight_layout()
vc.save(fig, "b1_density_plots.png")