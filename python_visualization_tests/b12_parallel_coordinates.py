# ============================================================
# B12 — Parallel coordinates: per-dataset standardised path of
#   each strategy family across 6 metrics (eps=0.01). Shows
#   multi-dimensional trade-offs (fast+low memory vs heavy).
# ============================================================
import viz_common as vc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = vc.load()
df = df[df["epsilon"] == 0.01].copy()
cols = ["render_mean_ms", "fps_mean", "gpu_mem_gb", "vol_samples",
        "dist_reads", "leaps", "skip_ratio_bounds"]

# Standardize so all axes share the same scale, invert "lower is better" elsewhere
Z = StandardScaler().fit_transform(df[cols].fillna(0))
D = pd.DataFrame(Z, columns=cols, index=df.index)
D["mode"] = df["mode"].values
D["fps_mean"] = -D["fps_mean"]                       # invert: high FPS -> low on axis
D["skip_ratio_bounds"] = -D["skip_ratio_bounds"]     # more skipping -> better -> low
# keep a modest sample so lines stay readable
D = D.sample(n=200, random_state=0)
D = D.sort_values("mode")

for c in cols:
    D[c] = (D[c] - D[c].min()) / (D[c].max() - D[c].min())

fig, ax = plt.subplots(figsize=(12, 7))
n = len(df["mode"].unique())
cmap = plt.cm.tab10(np.linspace(0, 1, n))
for i, (mode, g) in enumerate(D.groupby("mode")):
    for _, r in g.iterrows():
        ax.plot(cols, r[cols].astype(float), color=cmap[i], alpha=0.25, lw=0.8)
    ax.plot([], [], color=cmap[i], label=mode)
ax.set_ylim(0, 1.05)
ax.set_ylabel("normalized rank scale (higher = worse for all axes)")
ax.tick_params(axis="x", rotation=45)
ax.set_title("Parallel coordinates (eps=0.01) — all axes inverted so lower = better")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
vc.save(fig, "b12_parallel_coordinates.png")