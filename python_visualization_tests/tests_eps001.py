# ============================================================
# Scatter plots for epsilon = 0.01 ONLY
# Two pairs of plots, each with a color legend:
#   - fps_mean vs gpu_mem_bytes (GB)
#   - fps_mean vs sparsity_pct
# epsilon == 0.01 contains every strategy family
# (manual, optix, adaptive, bricked-regions, nanovdb,
#  octree, octree-regions).
# ============================================================
import re
import io

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "/home/adri/projects/TFM/optix_test_results/results.csv"

raw = open(CSV_PATH).read()
fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
df = pd.read_csv(io.StringIO(fixed))

df = df[df["epsilon"] == 0.01]
df["gpu_mem_gb"] = df["gpu_mem_bytes"] / 1e9

print(f"Rows with epsilon=0.01: {len(df)}")
print(f"Modes present: {sorted(df['mode'].unique())}\n")


def scatter_with_legend(x_col, xlabel, title, out_png):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x=x_col, y="fps_mean", hue="mode",
                    hue_order=sorted(df["mode"].unique()), ax=ax, alpha=0.8, s=40)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fps_mean")
    ax.set_title(title)
    ax.legend(title="mode", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_png}")


scatter_with_legend("gpu_mem_gb", "gpu_mem_bytes (GB)", "FPS vs GPU memory (epsilon = 0.01)",
                    "/home/adri/projects/TFM/optix_clone/python_visualization_tests/scatter_eps001_fps_vs_gpu.png")
scatter_with_legend("sparsity_pct", "sparsity (%)", "FPS vs sparsity (epsilon = 0.01)",
                    "/home/adri/projects/TFM/optix_clone/python_visualization_tests/scatter_eps001_fps_vs_sparsity.png")