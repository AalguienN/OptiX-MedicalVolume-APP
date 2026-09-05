# ============================================================
# D6 — Fidelity per STRATEGY, one panel per epsilon value.
#   TWO figures:
#     d6_luminance_per_epsilon.png  -> PSNR / SSIM / MAE (grayscale)
#     d6_rgb_per_epsilon.png        -> PSNR / SSIM / MAE (per-channel RGB avg)
#   Each cell = boxplot of the 47 datasets per strategy, coloured
#   by family, sorted by median (best fidelity on the left).
#   optix is pixel-identical to manual (PSNR = inf), shown at the
#   top; the PSNR axis is capped at 90 dB for readability.
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import viz_common as vc

CSV = "/home/adri/projects/TFM/optix_test_results/image_fidelity_metrics.csv"
fid = pd.read_csv(CSV)
fid = fid[fid["mode"] != "manual"]                  # manual-vs-manual is trivial
fid["psnr_gs"] = fid["psnr_gs"].clip(upper=90)      # keep optix(inf) on a readable axis
fid["psnr_rgb"] = fid["psnr_rgb"].clip(upper=90)

epsilons = sorted(fid["epsilon"].unique())
modes = sorted(fid["mode"].unique())
mode_color = {m: plt.cm.tab10(i) for i, m in enumerate(modes)}


def style(ax, col, eps):
    sub = fid[fid["epsilon"] == eps]
    order = sub.groupby("strategy")[col].median().sort_values(ascending=False).index
    pal = {s: mode_color[sub[sub["strategy"] == s]["mode"].iloc[0]] for s in order}
    sns.boxplot(data=sub, x="strategy", y=col, order=order, hue="strategy",
                palette=pal, ax=ax, fliersize=2, width=0.7, legend=False)
    ax.set_title(f"epsilon = {eps}", fontsize=10)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.set_xlabel("")
    if col.startswith("psnr"):
        ax.set_ylim(top=90)
    return order


def draw(metric_rows, title, out):
    nrows = len(metric_rows)
    fig, axes = plt.subplots(nrows, len(epsilons),
                             figsize=(4.0 * len(epsilons) + 1, 2.6 * nrows + 1.6))
    for j, (col, ylabel) in enumerate(metric_rows):
        for i, eps in enumerate(epsilons):
            ax = axes[j][i]
            order = style(ax, col, eps)
            if i > 0:
                ax.set_ylabel("")
        axes[j][0].set_ylabel(ylabel, fontsize=9)
    handles = [mpatches.Patch(color=mode_color[m], label=m) for m in modes]
    fig.legend(handles=handles, loc="upper center", ncol=len(modes),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    vc.save(fig, out)


draw([("psnr_gs", "PSNR luminance (dB)"),
      ("ssim_gs", "SSIM luminance"),
      ("mae_gs", "MAE luminance")],
     "Grayscale-luminance fidelity per strategy (47 datasets each) — PSNR capped at 90 dB",
     "d6_luminance_per_epsilon.png")

draw([("psnr_rgb", "PSNR RGB avg (dB)"),
      ("ssim_rgb", "SSIM RGB avg"),
      ("mae_rgb", "MAE RGB avg")],
     "Per-channel RGB fidelity per strategy (47 datasets each) — PSNR capped at 90 dB",
     "d6_rgb_per_epsilon.png")

print("=== Median grayscale PSNR per (strategy, epsilon) ===")
tab = fid.pivot_table(index="strategy", columns="epsilon", values="psnr_gs", aggfunc="median")
print(tab.round(2).fillna("-").to_string())

print("\n=== Median luminance SSIM per (strategy, epsilon) ===")
tab = fid.pivot_table(index="strategy", columns="epsilon", values="ssim_gs", aggfunc="median")
print(tab.round(4).fillna("-").to_string())