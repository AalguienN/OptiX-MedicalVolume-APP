# ============================================================
# A12 — Unsupervised exploration: standardize numeric features,
#   project to 2D with PCA, then see if KMeans finds clusters
#   that match the strategy families. Full ML pipeline demo.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = vc.load()
cols = ["render_mean_ms", "fps_mean", "gpu_mem_gb", "vol_samples",
        "dist_reads", "leaps", "skip_ratio_bounds", "sparsity_pct"]
X = StandardScaler().fit_transform(df[cols])

pca = PCA(n_components=2)
Z = pca.fit_transform(X)
print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.0%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.0%}")

df["pc1"], df["pc2"] = Z[:, 0], Z[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="mode", ax=axes[0], s=28, alpha=0.85)
axes[0].set_title("PCA of numeric features, colored by strategy family")
axes[0].legend(fontsize=8, bbox_to_anchor=(1, 1), loc="upper left")

km = KMeans(n_clusters=6, n_init=10, random_state=0)
df["cluster"] = km.fit_predict(X)
sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", palette="Set2", ax=axes[1], s=28, alpha=0.85)
axes[1].set_title("Same points, colored by KMeans cluster (k=6)")
axes[1].legend(title="cluster", fontsize=8)

fig.tight_layout()
vc.save(fig, "a12_pca_kmeans.png")

tab = df.pivot_table(index="cluster", columns="mode", values="dataset", aggfunc="count").fillna(0).astype(int)
print("\n=== Mode composition of each KMeans cluster ===")
print(tab.to_string())