# ============================================================
# B6 — Robust statistics: mean vs median vs trimmed mean vs
#   winsorized mean in a dot/strip chart. Shows how heavy the
#   tails are for each family (sensitivity to outliers).
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

df = vc.load()
y = "render_mean_ms"

fig, ax = plt.subplots(figsize=(13, 6.5))
order = sorted(df["mode"].unique())
sns.stripplot(data=df, x="mode", y=y, order=order, color="gray",
              alpha=0.25, size=2.5, jitter=0.25, zorder=1)

mean_y, med_y, tri_y, cli_y, xs = [], [], [], [], []
for i, mode in enumerate(order):
    g = df.loc[df["mode"] == mode, y]
    mean_y.append(g.mean())
    med_y.append(g.median())
    tri_y.append(trim_mean(g, 0.1))          # 10% trimmed mean
    cli_y.append(vc.clip_col(g).mean())      # winsorized mean

ax.scatter(np.arange(len(order)), mean_y, marker="^", s=70, color="crimson", label="mean", zorder=3)
ax.scatter(np.arange(len(order)) - 0.15, med_y, marker="o", s=60, color="navy", label="median", zorder=3)
ax.scatter(np.arange(len(order)) + 0.15, tri_y, marker="s", s=55, color="green", label="10% trimmed", zorder=3)
ax.set_yscale("log")
ax.set_ylabel("render_mean_ms (log)")
ax.set_title("Mean vs median vs trimmed mean per family — gap = sensitivity to outliers")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
fig.tight_layout()
vc.save(fig, "b6_robust_statistics.png")

print("=== mean / median / trimmed-mean per mode ===")
for i, mode in enumerate(order):
    print(f"  {mode:18s} mean={mean_y[i]:7.2f}  median={med_y[i]:7.2f}  "
          f"trimmed={tri_y[i]:7.2f}  winsorized={cli_y[i]:7.2f}")