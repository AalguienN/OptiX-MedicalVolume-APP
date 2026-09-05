# ============================================================
# A4 — Epsilon sweep: how do the adaptive strategies behave
#   as the error tolerance grows 0.01 -> 0.5?
#   Lines = mean across the 47 datasets, per strategy family.
# ============================================================
import viz_common as vc
import matplotlib.pyplot as plt

df = vc.load()
modes = [m for m in df["mode"].unique() if m != "manual"]
modes = [m for m in modes if m != "optix"]
agg = df[df["mode"].isin(modes)].groupby(["mode", "epsilon"]).agg(
    fps=("fps_mean", "mean"),
    gpu=("gpu_mem_gb", "mean"),
    render=("render_mean_ms", "mean"),
    skip=("skip_ratio_bounds", "mean"),
).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for mode in modes:
    g = agg[agg["mode"] == mode]
    axes[0, 0].plot(g["epsilon"], g["render"], marker="o", label=mode)
    axes[0, 1].plot(g["epsilon"], g["fps"], marker="o", label=mode)
    axes[1, 0].plot(g["epsilon"], g["gpu"], marker="o", label=mode)
    axes[1, 1].plot(g["epsilon"], g["skip"] * 100, marker="o", label=mode)

axes[0, 0].set_title("Render time (ms) vs epsilon");      axes[0, 0].set_xscale("log")
axes[0, 1].set_title("FPS vs epsilon");                    axes[0, 1].set_xscale("log")
axes[1, 0].set_title("GPU memory (GB) vs epsilon");        axes[1, 0].set_xscale("log")
axes[1, 1].set_title("Bounds skip ratio (%) vs epsilon");  axes[1, 1].set_xscale("log")
for ax in axes.ravel():
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.tight_layout()
vc.save(fig, "a4_epsilon_sweep.png")

tab = agg.pivot(index="mode", columns="epsilon", values="fps").round(0)
print("=== Mean FPS per (mode, epsilon) ===")
print(tab.to_string())