# ============================================================
# A6 — Dataset scaling: do progressively bigger/busier volumes
#   cost linearly more? Full renders (manual) act as the
#   reference "no acceleration" curve.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
full = df[df["mode"] == "manual"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for mode, g in df.groupby("mode"):
    n = 1 if mode == "manual" else 0.35
    alpha = 0.9 if mode == "manual" else 0.15
    axes[0].plot(g["num_slices"], g["render_mean_ms"], ".", ms=4,
                 alpha=alpha, label=mode if mode == "manual" else None)
    axes[1].plot(g["total_steps_bounds"] / 1e8, g["render_mean_ms"], ".", ms=4,
                 alpha=alpha, label=mode if mode == "manual" else None)

for ax, xl in zip(axes, ["num_slices", "total_steps_bounds / 1e8"]):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("render_mean_ms (log)")
    ax.set_xlabel(xl + " (log)")
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_title("Full-sampling render time vs volume slices")
axes[1].set_title("Render time vs total bounds steps (workload proxy)")
fig.tight_layout()
vc.save(fig, "a6_dataset_scaling.png")

# slope of log-log fit for manual = "how much worse per doubling of size"
from numpy.polynomial import polynomial as P
for col, name in [(full["num_slices"], "slices"), (full["total_steps_bounds"], "bounds steps")]:
    x, y = np.log(col.values), np.log(full["render_mean_ms"].values)
    slope, intercept = P.polyfit(x, y, 1)
    print(f"manual log-log slope vs {name}: {slope:.3f} " +
          f"(1.0 = linear cost, >1 = superlinear, <1 = sublinear)")