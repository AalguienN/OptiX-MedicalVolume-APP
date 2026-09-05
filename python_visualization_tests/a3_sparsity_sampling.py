# ============================================================
# A3 — Sparsity: do sparse volumes actually cost less?
#   LHS: volume samples needed vs sparsity % (log-y).
#   RHS: render time vs sparsity, per strategy family.
#   + Spearman rank correlation printed per mode.
# ============================================================
import viz_common as vc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = vc.load()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x="sparsity_pct", y="vol_samples", hue="mode",
                alpha=0.6, s=22, ax=axes[0], legend=False)
axes[0].set_yscale("log")
axes[0].set_title("Volume samples needed vs dataset sparsity (log-y)")

sns.scatterplot(data=df, x="sparsity_pct", y="render_mean_ms", hue="mode",
                alpha=0.55, s=22, ax=axes[1])
axes[1].set_title("Render time vs dataset sparsity, colored by family")
axes[1].legend(title="mode", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

fig.tight_layout()
vc.save(fig, "a3_sparsity_sampling.png")

for mode, g in df.groupby("mode"):
    rho, p = stats.spearmanr(g["sparsity_pct"], g["vol_samples"])
    rt, pt = stats.spearmanr(g["sparsity_pct"], g["render_mean_ms"])
    print(f"{mode:18s} sparsity~vol_samples rho={rho:+.2f} (p={p:.1e}) | "
          f"sparsity~render_time rho={rt:+.2f} (p={pt:.1e})")