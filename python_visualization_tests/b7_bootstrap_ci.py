# ============================================================
# B7 — Bootstrap confidence intervals for the median FPS of
#   each family. 10k resamples per family, percentile CI.
#   Robust way to rank families without normality assumptions.
# ============================================================
import viz_common as vc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
df = vc.load()

rows = []
for mode, g in df.groupby("mode"):
    vals = g["fps_mean"].values
    meds = np.percentile(rng.choice(vals, size=(5000, len(vals)), replace=True), 50, axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    rows.append({"mode": mode, "median_fps": np.median(vals), "ci_lo": lo, "ci_hi": hi})
tab = pd.DataFrame(rows).sort_values("median_fps", ascending=False)
print("=== Median FPS with 95% bootstrap CI (5000 resamples) ===")
print(tab.round(0).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
order = tab["mode"].tolist()
x = np.arange(len(order))
ax.barh(x, tab["median_fps"], xerr=[tab["median_fps"] - tab["ci_lo"], tab["ci_hi"] - tab["median_fps"]],
        color="steelblue", alpha=0.85, capsize=4)
ax.set_yticks(x); ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("median FPS (bar = 95% bootstrap CI)")
ax.set_title("Bootstrap median FPS per family — non-overlap of CIs ≈ significant")
fig.tight_layout()
vc.save(fig, "b7_bootstrap_ci.png")