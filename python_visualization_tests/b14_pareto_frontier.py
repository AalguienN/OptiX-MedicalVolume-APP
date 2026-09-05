# ============================================================
# B14 — Efficiency frontier: where is each strategy ON the
#   "cost vs speed" trade-off? Scatter of GPU memory vs render
#   time with a lower-convex-hull-ish baseline = the Pareto
#   frontier (lowest memory at each speed). eps=0.01 only.
# ============================================================
import viz_common as vc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
df = df[df["epsilon"] == 0.01].copy()

# Pareto frontier for MINIMISING both axes (memory, time)
pts = df[["gpu_mem_gb", "render_mean_ms"]].values
order = np.argsort(pts[:, 0])
front = []
best_time = np.inf
for i in order:
    mem, t = pts[i]
    if t < best_time:                      # strictly better on time than anything left
        best_time = t
        front.append((mem, t))
front = np.array(front)

fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(data=df, x="gpu_mem_gb", y="render_mean_ms", hue="mode",
                alpha=0.6, s=26, ax=ax, legend=False)
ax.plot(front[:, 0], front[:, 1], "k--", lw=1.8, label="Pareto frontier (less is better)")
ax.set_yscale("log")
ax.set_xlabel("gpu_mem_gb")
ax.set_ylabel("render_mean_ms (log)")
ax.set_title("Efficiency frontier — memory could be sacrificed? (eps=0.01)")
ax.legend(fontsize=9)
fig.tight_layout()
vc.save(fig, "b14_pareto_frontier.png")

# who's ON the frontier?
front_key = set(zip(front[:, 0].round(6), front[:, 1].round(6)))
df["front_key"] = list(zip(df["gpu_mem_gb"].round(6), df["render_mean_ms"].round(6)))
names = df.loc[df["front_key"].isin(front_key), ["mode", "strategy", "gpu_mem_gb", "render_mean_ms"]].sort_values("render_mean_ms")
print("=== Points lying on the Pareto frontier ===")
print(names.to_string(index=False))