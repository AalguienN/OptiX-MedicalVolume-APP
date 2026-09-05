# ============================================================
# B3 — Hexbin plots: where do the points concentrate in each
#   2-D metric pair? Colourbar = number of points per hexbin.
# ============================================================
import viz_common as vc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

df = vc.load()

def hexbin(ax, xcol, ycol, logx, logy):
    x, y = df[xcol].values, df[ycol].values
    hb = ax.hexbin(x, y, gridsize=35, cmap="viridis", mincnt=1, norm=LogNorm())
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    cb = ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("count")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
hexbin(axes[0], "gpu_mem_gb", "fps_mean", logx=False, logy=True)
axes[0].set_title("FPS vs GPU memory (log-y)")
hexbin(axes[1], "sparsity_pct", "render_mean_ms", logx=False, logy=True)
axes[1].set_title("Render time vs sparsity (log-y)")
hexbin(axes[2], "vol_samples", "render_mean_ms", logx=True, logy=True)
axes[2].set_title("Render time vs volume samples (log-log)")

fig.tight_layout()
vc.save(fig, "b3_hexbin_jointplots.png")