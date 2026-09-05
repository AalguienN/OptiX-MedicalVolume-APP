# ============================================================
# B5 — Binning by sparsity deciles: is the sparsity penalty
#   smooth or step-like? Mean render time per decile bucket,
#   one line per strategy family (log-y to see differences).
# ============================================================
import viz_common as vc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = vc.load()
df["sparsity_bin"] = pd.qcut(df["sparsity_pct"], 10, labels=False)
agg = df.groupby(["sparsity_bin", "mode"])["render_mean_ms"].mean().reset_index()
bin_edges = df.groupby("sparsity_bin")["sparsity_pct"].mean()

fig, ax = plt.subplots(figsize=(12, 6.5))
for mode, g in agg.groupby("mode"):
    y = g.set_index("sparsity_bin")["render_mean_ms"]
    y = y.reindex(range(10))
    ax.plot(bin_edges.reindex(y.index).values, y, marker="o", lw=1.5, label=mode)
ax.set_yscale("log")
ax.set_xlabel("mean sparsity (%) of decile bucket")
ax.set_ylabel("mean render time (ms, log)")
ax.set_title("Mean render time per sparsity decile (Q10 quantization)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
vc.save(fig, "b5_sparsity_binning.png")

print("=== Spearman between sparsity decile and mean render time ===")
for mode, g in agg.groupby("mode"):
    rho = g["sparsity_bin"].corr(g["render_mean_ms"], method="spearman")
    print(f"  {mode:18s} rho={rho:+.2f}")