# ============================================================
# D5 — Is faster cheaper in image quality too? PSNR (grayscale)
#   vs skip_ratio_bounds and vs speedup over manual. If the
#   relation is one-sided, part of the FPS gain is paid in
#   fidelity — this plot shows exactly how much.
# ============================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import viz_common as vc

FID = "/home/adri/projects/TFM/optix_test_results/image_fidelity_metrics.csv"
df = vc.load()
f = pd.read_csv(FID)
df = df.merge(f, on=["dataset", "strategy", "mode", "epsilon"])
df = df[df["mode"] != "manual"].copy()

base = df[df["mode"] == "optix"].groupby("dataset")["render_mean_ms"].median()
df["speedup_vs_optix"] = df.apply(lambda r: base[r["dataset"]] / r["render_mean_ms"], axis=1)

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

sns.scatterplot(data=df, x="skip_ratio_bounds", y="psnr_gs", hue="mode",
                alpha=0.4, s=15, ax=axes[0], legend=False)
axes[0].set_title("PSNR vs bound-skip ratio (more skip = ?)")

sns.scatterplot(data=df, x="speedup_vs_optix", y="psnr_gs", hue="mode",
                alpha=0.4, s=15, ax=axes[1])
axes[1].set_xscale("log")
axes[1].set_title("PSNR vs speedup over OptiX (log-x, >1 = faster)")
axes[1].legend(title="mode", fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

sns.boxplot(data=df, x="mode", y="psnr_gs", order=sorted(df["mode"].unique()),
            ax=axes[2], fliersize=2)
axes[2].set_yscale("log")
axes[2].tick_params(axis="x", rotation=45)
axes[2].set_title("PSNR across ALL epsilons (log-y)")

fig.tight_layout()
vc.save(fig, "d5_fidelity_vs_speed.png")

r1, p1 = stats.spearmanr(df["skip_ratio_bounds"], df["psnr_gs"])
r2, p2 = stats.spearmanr(df["speedup_vs_optix"], df["psnr_gs"])
print(f"PSNR ~ skip_ratio_bounds:   rho={r1:+.3f} (p={p1:.1e})")
print(f"PSNR ~ speedup_vs_optix:    rho={r2:+.3f} (p={p2:.1e})")

print("\n=== Mean PSNR per family across all epsilons ===")
print(df.groupby("mode")["psnr_gs"].agg(["mean", "median", "min"]).round(2).to_string())