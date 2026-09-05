# ============================================================
# D3 — Fidelity vs epsilon: quality cost of the error
#   tolerance. Mean PSNR and SSIM (over the 47 datasets) as
#   epsilon grows 0.01 -> 0.50, one line per family.
# ============================================================
import matplotlib.pyplot as plt
import pandas as pd
import viz_common as vc

CSV = "/home/adri/projects/TFM/optix_test_results/image_fidelity_metrics.csv"
df = pd.read_csv(CSV)
df = df[df["mode"] != "manual"].copy()
modes = sorted(df["mode"].unique())
epsilons = sorted(df["epsilon"].unique())

agg = df.groupby(["mode", "epsilon"])[
    ["psnr_gs", "psnr_rgb", "ssim_gs", "ssim_rgb"]].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for mode in modes:
    g = agg[agg["mode"] == mode]
    axes[0].plot(g["epsilon"], g["psnr_gs"], marker="o", lw=1.5, label=mode)
    axes[1].plot(g["epsilon"], g["ssim_gs"], marker="o", lw=1.5, label=mode)
axes[0].set_xscale("log"); axes[1].set_xscale("log")
axes[0].set_title("Mean PSNR (grayscale) vs epsilon log-x"); axes[0].set_ylabel("PSNR (dB)")
axes[1].set_title("Mean SSIM (grayscale) vs epsilon log-x"); axes[1].set_ylabel("SSIM")
for ax in axes:
    ax.set_xlabel("epsilon")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
vc.save(fig, "d3_fidelity_vs_epsilon.png")

print("=== Mean grayscale PSNR per (mode, epsilon) ===")
print(agg.pivot(index="mode", columns="epsilon", values="psnr_gs").round(2).to_string())