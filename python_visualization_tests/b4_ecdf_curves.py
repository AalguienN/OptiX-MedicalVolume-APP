# ============================================================
# B4 — ECDF curves: the probability a strategy finishes a
#   dataset in under X ms (or runs faster than Y FPS). Step
#   curves highlight where one family dominates another.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
metrics = [("render_mean_ms", "render time (ms, log-x)"), ("fps_mean", "FPS (log-x)")]
for ax, (col, label) in zip(axes, metrics):
    for mode, g in df.groupby("mode"):
        x = np.sort(g[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, marker=".", ms=4, label=mode, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(label); ax.set_ylabel("cumulative fraction of datasets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_title("ECDF — fraction of datasets rendered within X ms")
axes[1].set_title("ECDF — fraction of datasets at or above X FPS")
fig.tight_layout()
vc.save(fig, "b4_ecdf_curves.png")

print("=== Fraction of datasets under 5 ms, per mode ===")
for mode, g in df.groupby("mode"):
    print(f"  {mode:18s} {(g['render_mean_ms'] < 5).mean()*100:5.1f}%")