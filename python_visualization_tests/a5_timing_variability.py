# ============================================================
# A5 — Timing consistency: how much the per-frame cost jitters
#   within a run. variability = (max - min) / mean, from the
#   200-frame render loop. Slow strategies tend to be jitterier.
# ============================================================
import viz_common as vc
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
df["variability"] = (df["render_max_ms"] - df["render_min_ms"]) / df["render_mean_ms"]
df = df[df["variability"].notna()]

fig, ax = plt.subplots(figsize=(13, 6))
order = sorted(df["mode"].unique())
sns.boxplot(data=df, x="mode", y="variability", order=order, ax=ax, fliersize=2)
ax.set_title("Frame-time variability (max-min)/mean — lower = more stable")
ax.set_ylabel("(render_max_ms - render_min_ms) / render_mean_ms")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
vc.save(fig, "a5_timing_variability.png")

print("=== Median variability per mode ===")
print(df.groupby("mode")["variability"].median().round(3).to_string())