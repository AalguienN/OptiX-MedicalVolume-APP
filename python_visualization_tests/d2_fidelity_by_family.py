# ============================================================
# D2 — Image fidelity by strategy family at epsilon = 0.01
#   (the only epsilon where every family ran). PSNR & SSIM,
#   grayscale vs per-channel RGB, all 47 datasets.
# ============================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import viz_common as vc

CSV = "/home/adri/projects/TFM/optix_test_results/image_fidelity_metrics.csv"
df = pd.read_csv(CSV)
df = df[df["epsilon"] == 0.01].copy()
df = df[df["mode"] != "manual"]          # manual vs manual is trivial (identity)

order = sorted(df["mode"].unique())
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

for ax, col, title in [
    (axes[0], "psnr_gs", "PSNR grayscale (dB)"),
    (axes[1], "psnr_rgb", "PSNR per-channel avg (dB)"),
    (axes[2], "ssim_gs", "SSIM grayscale"),
    (axes[3], "ssim_rgb", "SSIM per-channel avg"),
]:
    sns.boxplot(data=df, x="mode", y=col, order=order, ax=ax, fliersize=2)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Image fidelity vs the manual reference per family (epsilon = 0.01)", fontsize=13)
fig.tight_layout()
vc.save(fig, "d2_fidelity_by_family.png")

print("=== Median PSNR/SSIM per family at eps=0.01 ===")
tab = df.groupby("mode")[["psnr_gs", "ssim_gs", "psnr_rgb", "ssim_rgb"]].median().round(3)
print(tab.to_string())