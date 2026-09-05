# ============================================================
# A2 — GPU memory trade-off: throughput per GB and the
#   memory overhead strategies pay vs the manual baseline.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
df["fps_per_gb"] = df["fps_mean"] / df["gpu_mem_gb"]

# Baseline memory: median manual gpu usage per dataset
base = df[df["mode"] == "manual"].groupby("dataset")["gpu_mem_gb"].median()
df["gpu_overhead"] = df.apply(lambda r: r["gpu_mem_gb"] / base[r["dataset"]] - 1, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

sns.scatterplot(data=df, x="gpu_mem_gb", y="fps_mean", hue="mode",
                alpha=0.6, s=24, ax=axes[0], legend=False)
axes[0].set_xscale("log")
axes[0].set_title("FPS vs GPU memory (log-x)")
axes[0].set_xlabel("gpu_mem_gb (log)")

order = sorted(df["mode"].unique())
sns.boxplot(data=df, x="mode", y="fps_per_gb", order=order, ax=axes[1], fliersize=2)
axes[1].set_title("Throughput per GB of GPU memory (FPS/GB)")
axes[1].tick_params(axis="x", rotation=45)

sns.boxplot(data=df[df["mode"] != "manual"], x="mode", y="gpu_overhead",
            order=[m for m in order if m != "manual"], ax=axes[2], fliersize=2)
axes[2].axhline(0, ls="--", color="crimson", lw=1.2)
axes[2].set_title("GPU memory overhead vs manual (0 = same, % )")
axes[2].tick_params(axis="x", rotation=45)

fig.tight_layout()
vc.save(fig, "a2_gpu_memory_tradeoff.png")