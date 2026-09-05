import re
import io

import pandas as pd          # Data tables / CSV loading
import numpy as np           # Numeric arrays (used via pandas)
import seaborn as sns        # Nice statistical plots
import matplotlib.pyplot as plt  # Low-level plotting backend
from scipy import stats      # Statistical tests (SciPy)
from sklearn.preprocessing import StandardScaler  # ML preprocessing

# ============================================================
# 2. LOAD THE CSV
#    the CSV stores tuples like (x,y,z) in `spacing` and
#    `origin` WITHOUT quotes, so the commas inside them split
#    the row into extra columns and shift everything right.
#    Fix: find (...<comma>...) groups and wrap them in double
#    quotes BEFORE pandas reads the file.
# ============================================================
CSV_PATH = "/home/adri/projects/TFM/optix_test_results/results.csv"

raw = open(CSV_PATH).read()                                   # Read file as plain text
fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)  # Quote "tuple,with,commas"
df = pd.read_csv(io.StringIO(fixed))                          # Parse the cleaned text into a DataFrame

# ============================================================
# 3. SANITY CHECK
# ============================================================
print("=== Parsed data (head) ===")
print(df[["strategy", "mode", "modality", "render_mean_ms", "fps_mean"]].head())
print(f"\n{len(df)} rows, {df['strategy'].nunique()} strategies\n")

# ============================================================
# 4. PANDAS SUMMARY TABLE
#    Group rows by `mode` (the strategy *family*, e.g. all the
#    bricked-regions variants collapse into one bucket) and
#    compute mean + std of the key timing/GPU metrics.
# ============================================================
key_cols = ["render_mean_ms", "fps_mean", "gpu_mem_bytes", "vol_samples"]
print("=== Summary by strategy family (mode) ===")
print(df.groupby("mode")[key_cols].agg(["mean", "std"]).round(2))
print()

# ============================================================
# 5. SEABORN BOXPLOTS
#    Two side-by-side boxplots on one figure:
#      - left:  render time per strategy family
#      - right: FPS per strategy family
#    `ax=axes[i]` tells each plot which subplot to draw into.
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))   # 1 row, 2 columns of subplots

sns.boxplot(x="mode", y="render_mean_ms", data=df, ax=axes[0])   # Left panel
axes[0].set_title("Render Mean Time (ms) by Strategy Family")
axes[0].tick_params(axis="x", rotation=45)                       # Rotate long labels

sns.boxplot(x="mode", y="fps_mean", data=df, ax=axes[1])         # Right panel
axes[1].set_title("Mean FPS by Strategy Family")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()                                               # Avoid overlapping labels
plt.savefig("/home/adri/projects/TFM/optix_clone/python_visualization_tests/strategy_comparison.png", dpi=150)  # Save instead of showing
print("Saved plot to python_visualization_tests/strategy_comparison.png")

# ============================================================
# 6. SCIPY STATISTICAL TEST
#    Welch's t-test (equal_var=False) compares the mean render
#    time of `manual` vs `optix`. Small p-value (< 0.05) would
#    mean the difference is statistically significant.
# ============================================================
manual = df.loc[df["mode"] == "manual", "render_mean_ms"]   # Filter rows, keep one column
optix = df.loc[df["mode"] == "optix", "render_mean_ms"]
t_stat, p_val = stats.ttest_ind(manual, optix, equal_var=False)
print(f"\nWelch's t-test manual vs optix render_mean_ms: t={t_stat:.3f}, p={p_val:.4f}")

# ============================================================
# 7. SCIKIT-LEARN STANDARDIZATION
#    Scale numeric features to mean=0, std=1 so columns with
#    different units (ms vs bytes vs %) can be compared/used
#    by ML models fairly.
# ============================================================
scale_cols = ["render_mean_ms", "fps_mean", "n_relevant", "sparsity_pct"]
X = StandardScaler().fit_transform(df[scale_cols].fillna(0))  # fit=learn mean/std, transform=apply
print("\n=== Standardized feature sample (first 5 rows) ===")
print(pd.DataFrame(X, columns=scale_cols).head())

# ============================================================
# 8. EPSILON-FACETED SCATTER PLOTS
#    One panel per epsilon value (0.01, 0.05, 0.10, 0.20, 0.50).
#    manual/optix do NOT use epsilon (0.01 is just a placeholder),
#    so they are injected as baselines into EVERY panel.
# ============================================================
epsilons = sorted(df["epsilon"].unique())                 # [0.01, 0.05, 0.10, 0.20, 0.50]
BASELINE_MODES = ["manual", "optix"]
baseline = df[df["mode"].isin(BASELINE_MODES)]            # 47 pts per baseline strategy
df["gpu_mem_gb"] = df["gpu_mem_bytes"] / 1e9              # bytes -> GB for a readable axis

# Keep baselines first in the legend/color order
hue_order = BASELINE_MODES + sorted(m for m in df["mode"].unique() if m not in BASELINE_MODES)


def make_epsilon_scatter(x_col, xlabel, title, out_png):
    """Build 1 row of scatter subplots (one per epsilon), all sharing the FPS y-axis."""
    fig, axes = plt.subplots(1, len(epsilons), figsize=(20, 4.5), sharey=True, squeeze=False)
    for ax, eps in zip(axes[0], epsilons):
        # Strategies that actually ran at this epsilon + the baselines
        subset = pd.concat([df[df["epsilon"] == eps], baseline]).drop_duplicates()
        sns.scatterplot(data=subset, x=x_col, y="fps_mean", hue="mode",
                        hue_order=hue_order, ax=ax, alpha=0.75, s=28, legend=False)
        ax.set_title(f"epsilon = {eps}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
    axes[0][0].set_ylabel("fps_mean")
    # Single shared legend instead of one per subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(hue_order),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"{title} (each point = one dataset)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


make_epsilon_scatter("gpu_mem_gb", "gpu_mem_bytes (GB)", "FPS vs GPU memory per epsilon",
                     "/home/adri/projects/TFM/optix_clone/python_visualization_tests/scatter_fps_vs_gpu.png")
make_epsilon_scatter("sparsity_pct", "sparsity (%)", "FPS vs sparsity per epsilon",
                     "/home/adri/projects/TFM/optix_clone/python_visualization_tests/scatter_fps_vs_sparsity.png")
