# ============================================================
# A7 — Modality: CT vs MR volumes. MR volumes differ in
#   contrast/sparsity; does that change the win of each
#   strategy family over manual?
# ============================================================
import viz_common as vc
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
sns.boxplot(data=df, x="mode", y="render_mean_ms", hue="modality", ax=axes[0], fliersize=1.5)
axes[0].set_title("Render time by modality")
sns.boxplot(data=df, x="mode", y="gpu_mem_gb", hue="modality", ax=axes[1], fliersize=1.5)
axes[1].set_title("GPU memory by modality")
for ax in axes:
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
vc.save(fig, "a7_modality_analysis.png")

print("=== Median render ms by (mode, modality) ===")
print(df.pivot_table(index="mode", columns="modality", values="render_mean_ms", aggfunc="median").round(3).to_string())