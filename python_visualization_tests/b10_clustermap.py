# ============================================================
# B10 — Clustermap of render time: strategy families ×
#   datasets, row-normalized (z-score per dataset) to reveal
#   WHICH families behave alike, independent of raw speed.
# ============================================================
import viz_common as vc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
df = df[df["epsilon"] == 0.01]
mat = df.pivot_table(index="mode", columns="dataset", values="render_mean_ms")

g = sns.clustermap(mat, z_score=1, cmap="viridis", figsize=(18, 5),
                   row_cluster=True, col_cluster=False, cbar_pos=(0.95, 0.35, 0.03, 0.3))
g.ax_heatmap.set_xlabel("datasets")
g.ax_heatmap.set_ylabel("")
g.figure.suptitle("Z-scored render time per dataset — families with similar colour profiles behave alike", y=1.02)
g.figure.tight_layout()
g.figure.savefig(vc.PLOT_DIR + "/b10_clustermap.png", dpi=150, bbox_inches="tight")
print("  saved " + vc.PLOT_DIR + "/b10_clustermap.png")

# similarity table: mean |z-difference| between families
z = mat.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
modes = z.index.tolist()
dist = pd.DataFrame(index=modes, columns=modes, dtype=float)
for a in modes:
    for b in modes:
        dist.loc[a, b] = (z.loc[a] - z.loc[b]).abs().mean()
print("\n=== Mean |z-score distance| between families (lower = more similar behaviour) ===")
print(dist.round(3).to_string())