# ============================================================
# B13 — Violin + modality split for the TOP 4 families:
#   shows the whole distribution shape (not just boxes) and
#   whether CT vs MR volumes change the picture.
# ============================================================
import viz_common as vc
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
top = (df.groupby("mode")["render_mean_ms"].median().sort_values().head(4).index.tolist())
sub = df[df["mode"].isin(top)]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.violinplot(data=sub, x="mode", y="render_mean_ms", hue="modality",
               split=True, density_norm="width", inner="quartile",
               order=top, ax=axes[0], cut=0)
axes[0].set_yscale("log")
axes[0].set_title("Top-4 families, CT vs MR, render time (split violin)")

sns.violinplot(data=sub, x="mode", y="fps_mean",
               hue="modality", split=True, density_norm="width", inner="quartile",
               order=top, ax=axes[1], cut=0)
axes[1].set_yscale("log")
axes[1].set_title("Top-4 families, CT vs MR, FPS (split violin)")

for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
fig.tight_layout()
vc.save(fig, "b13_violin_top4_modality.png")