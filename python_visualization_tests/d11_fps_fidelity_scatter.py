# ============================================================
# D11 — Per-strategy scatter: FPS vs visual fidelity, colored by epsilon
#   Uses ALL runs from ../optix_test_results2/results.csv (4042 rows).
#   One group per strategy configuration (manual, optix, and every
#   brick/leaf/sampler variant across all 4 epsilons).
#   X  = fps_mean (from results.csv)
#   Y  = fidelity metric vs the dataset's manual reference
#   c  = epsilon (colorbar)
#
#   Emits ONE figure per metric:
#     d11_fps_fidelity_psnr.png / _ssim.png / _mae.png / _pct_changed.png
#
#   Per-snapshot metrics are cached in plots/d11_fidelity_cache.csv so
#   later runs only (re)compute missing images.
#
#   Usage: venv/bin/python d11_fps_fidelity_scatter.py
# ============================================================
import re
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_ROOT = Path("/home/adri/projects/TFM/optix_test_results2")
RESULTS_CSV = RESULTS_ROOT / "results.csv"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)
CACHE_CSV = PLOT_DIR / "d11_fidelity_cache.csv"

EPSILONS = [0.01, 0.05, 0.1, 0.2]

METRICS = [
    ("psnr",        "PSNR (dB)",      "higher is better"),
    ("ssim",        "SSIM",           "higher is better"),
    ("mae",         "MAE (0-255)",    "lower is better"),
    ("pct_changed", "% changed pixels", "lower is better"),
]

# deterministic ordering of strategy families
MODE_ORDER = ["manual", "optix", "adaptive",
              "bricked-regions", "octree", "octree-regions", "nanovdb"]


# ── strategy naming ──────────────────────────────────────────
def parse_strategy(s):
    """Parse a results.csv 'strategy' label -> (mode, eps_code, params).

    manual / optix -> (s, None, {}); sweep like
    'bricked-regions_eps010-brick16-amoff' -> (mode, 'eps010', {brick,am}).
    """
    if s in ("manual", "optix"):
        return s, None, {}
    m = re.match(r"^(.*)_(eps\d{3})((?:-[^-]+)*)$", s)
    if not m:
        return s, None, {}
    mode, eps_code = m.group(1), m.group(2)
    params = {}
    for part in m.group(3).lstrip("-").split("-"):
        if part.startswith("brick"):
            params["brick"] = part[5:]
        elif part.startswith("leaf"):
            params["leaf"] = part[4:]
        elif part.startswith("am"):
            params["am"] = part[2:]
        elif part in ("nearest", "trilinear"):
            params["sampler"] = part
    return mode, eps_code, params


def group_label(mode, params):
    if not params:
        return mode
    parts = []
    if "brick" in params:
        parts.append(f"brick{params['brick']}")
    if "leaf" in params:
        parts.append(f"leaf{params['leaf']}")
    if "sampler" in params:
        parts.append(params["sampler"])
    if "am" in params:
        parts.append(f"am-{params['am']}")
    return f"{mode} | {' '.join(parts)}"


def group_key(mode, params):
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else 99, mode,
            frozenset(params.items()))


# ── image processing ─────────────────────────────────────────
def _gray(a):
    return 0.299 * a[..., 0].astype(np.float64) + \
           0.587 * a[..., 1].astype(np.float64) + \
           0.114 * a[..., 2].astype(np.float64)


def foreground_mask(ref, threshold=5):
    return np.any(ref > threshold, axis=2)


def compute_metrics(img, ref, mask, ssim_scale=4):
    """Foreground-masked PSNR/MAE/%changed + SSIM (downscaled luminance)."""
    fimg = img[mask].astype(np.float64)
    fref = ref[mask].astype(np.float64)

    mae = np.mean(np.abs(fimg - fref))
    mse = np.mean((fimg - fref) ** 2)
    psnr = 100.0 if mse < 1e-10 else 10.0 * np.log10(255.0 ** 2 / mse)
    diff = np.abs(fimg - fref)
    pct_changed = 100.0 * np.any(diff > 2.0, axis=1).sum() / max(diff.shape[0], 1)

    # SSIM on downscaled luminance (cheap; negligible value shift)
    def dec(a, interp):
        return np.asarray(
            Image.fromarray(a).resize(
                (a.shape[1] // ssim_scale, a.shape[0] // ssim_scale), interp
            ),
            dtype=np.float64,
        )
    li = dec(_gray(img), Image.LANCZOS)
    lr = dec(_gray(ref), Image.LANCZOS)
    lm = dec(mask.astype(np.uint8) * 255, Image.NEAREST).astype(np.bool_)
    sig = 1.5
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mi = gaussian_filter(li, sig)
    mr = gaussian_filter(lr, sig)
    si = gaussian_filter(li ** 2, sig) - mi ** 2
    sr = gaussian_filter(lr ** 2, sig) - mr ** 2
    sc = gaussian_filter(li * lr, sig) - mi * mr
    smap = ((2 * mi * mr + C1) * (2 * sc + C2)) / \
           ((mi ** 2 + mr ** 2 + C1) * (si + sr + C2))
    ssim = float(np.mean(smap[lm])) if lm.any() else float("nan")

    return {"psnr": psnr, "ssim": ssim, "mae": mae, "pct_changed": pct_changed}


# ── metrics cache ────────────────────────────────────────────
def load_cache():
    if CACHE_CSV.exists():
        c = pd.read_csv(CACHE_CSV)
        return {r["snapshot"]: r for _, r in c.iterrows()}
    return {}


def save_cache(cache):
    pd.DataFrame.from_dict(cache, orient="index").to_csv(CACHE_CSV, index=False)


# ── main ─────────────────────────────────────────────────────
def main():
    raw = open(RESULTS_CSV).read()
    fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
    df = pd.read_csv(io.StringIO(fixed))
    print(f"{len(df)} rows loaded across {df['dataset'].nunique()} datasets")

    cache = load_cache()
    groups = {}                       # sorted group index -> (label, mode, params)

    # ── compute/collect metrics per row ──────────────────────
    refs = {}
    recs = []
    todo_saved = False
    for n, (_, r) in enumerate(df.iterrows(), 1):
        snap = r["snapshot_path"]
        if not Path(snap).exists():
            continue
        if snap in cache:
            m = cache[snap]
        else:
            ddir = Path(snap).parent
            if ddir not in refs:
                man_img = np.asarray(Image.open(ddir / "manual.ppm"), dtype=np.uint8)
                refs[ddir] = (man_img, foreground_mask(man_img))
            img = np.asarray(Image.open(snap), dtype=np.uint8)
            m = compute_metrics(img, *refs[ddir])
            cache[snap] = {
                "snapshot": snap,
                "psnr": m["psnr"], "ssim": m["ssim"],
                "mae": m["mae"], "pct_changed": m["pct_changed"],
            }
            todo_saved = True
            if n % 250 == 0:
                print(f"  computed {n}/{len(df)}  (new: {len(cache)})")
        mode, _eps_code, params = parse_strategy(r["strategy"])
        key = group_key(mode, params)
        if key not in groups:
            groups[key] = (group_label(mode, params), mode, params)
        recs.append({
            "group": key,
            "fps": r["fps_mean"],
            "eps": r["epsilon"],
            "psnr": m["psnr"], "ssim": m["ssim"],
            "mae": m["mae"], "pct_changed": m["pct_changed"],
        })
    if todo_saved:
        save_cache(cache)
        print(f"  cached {len(cache)} snapshot metrics -> {CACHE_CSV}")

    order = sorted(groups)
    g_labels = [groups[k][0] for k in order]
    n_groups = len(order)
    print(f"  {n_groups} strategy groups, {len(recs)} points")

    # ── one figure per metric ────────────────────────────────
    norm = mcolors.Normalize(vmin=0.0, vmax=0.2)
    cmap = plt.cm.viridis

    n_cols = 7
    n_rows = int(np.ceil(n_groups / n_cols))
    for mname, mlabel, mnote in METRICS:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.7 * n_cols, 2.4 * n_rows),
                                 squeeze=False)
        for gi, key in enumerate(order):
            ax = axes[gi // n_cols, gi % n_cols]
            sel = [rec for rec in recs if rec["group"] == key]
            if sel:
                q = np.random.default_rng(0).permutation(len(sel))  # paint order
                sel = [sel[i] for i in q]
                xs = np.array([rec["fps"] for rec in sel])
                ys = np.array([rec[mname] for rec in sel])
                es = np.array([rec["eps"] for rec in sel])
                ax.scatter(xs, ys, c=es, cmap=cmap, norm=norm,
                           s=14, alpha=0.7, edgecolors="none")
            ax.set_title(g_labels[gi], fontsize=6.5)
            ax.grid(True, alpha=0.3, linewidth=0.4)
            ax.tick_params(labelsize=5.5)
            if gi % n_cols == 0:
                ax.set_ylabel(f"{mlabel}\n({mnote})", fontsize=6.5)
            if gi // n_cols == n_rows - 1:
                ax.set_xlabel("FPS", fontsize=6.5)
        for gi in range(n_groups, n_rows * n_cols):
            axes[gi // n_cols, gi % n_cols].axis("off")

        fig.suptitle(f"{mlabel} vs FPS per strategy — all {len(df)} runs "
                     f"({n_groups} groups)\ncolor = epsilon · "
                     "foreground-masked, manual reference",
                     fontsize=11)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=axes, ticks=EPSILONS, pad=0.01, shrink=0.85)
        cbar.set_label("epsilon", fontsize=8)
        cbar.ax.tick_params(labelsize=6)
        fig.subplots_adjust(left=0.09, right=0.93, top=0.88, bottom=0.07,
                            wspace=0.30, hspace=0.45)
        out = PLOT_DIR / f"d11_fps_fidelity_{mname}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()