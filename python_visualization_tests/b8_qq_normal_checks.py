# ============================================================
# B8 — QQ plots + fitted normal histogram for render time:
#   are the tails normal-ish? Log-transform collapses the
#   right skew. Shows why median/robust stats > mean here.
# ============================================================
import viz_common as vc
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

df = vc.load()

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for i, mode in enumerate(["manual", "optix", "octree", "adaptive"]):
    y = np.log(df.loc[df["mode"] == mode, "render_mean_ms"].values)
    ax = axes.flat[i]
    st.probplot(y, dist="norm", plot=ax)
    ax.set_title(f"{mode} — QQ plot of log render time")
    ax.get_lines()[1].set_color("crimson")

fig.tight_layout()
vc.save(fig, "b8_qq_normal_checks.png")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for i, mode in enumerate(["manual", "optix", "octree", "adaptive"]):
    y = np.log(df.loc[df["mode"] == mode, "render_mean_ms"].values)
    ax = axes.flat[i]
    mu, sd = st.norm.fit(y)
    ax.hist(y, bins=25, density=True, alpha=0.6, color="steelblue")
    xs = np.linspace(y.min(), y.max(), 200)
    ax.plot(xs, st.norm.pdf(xs, mu, sd), "r-", lw=1.5, label="fitted normal")
    ax.set_title(f"{mode}: log-render ~ N({mu:.2f}, {sd:.2f})")
    ax.legend(fontsize=8)
fig.tight_layout()
vc.save(fig, "b8b_histogram_normal_fit.png")