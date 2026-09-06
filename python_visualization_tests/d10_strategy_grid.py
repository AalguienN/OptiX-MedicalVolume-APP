# ============================================================
# D10 — Simplified test-matrix image grid with quality metrics
#   One row per strategy, epsilon sweep across columns.
#   For multi-param strategies picks the MIDDLE variant:
#     bricked-regions → brick16, am-off
#     octree          → leaf8,  am-off
#     octree-regions  → leaf8,  am-off
#     nanovdb         → nearest
#
#   Metrics (foreground-masked): PSNR, SSIM, MAE, % changed px
#
#   Usage: venv/bin/python d10_strategy_grid.py [dataset]
# ============================================================
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "optix_test_results2"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

EPSILONS = [0.01, 0.05, 0.1, 0.2]
EPS_CODES = ["eps001", "eps005", "eps010", "eps020"]

# ── grid: one row per strategy, middle-variant picks ─────────
# (label, mode, eps_code -> frozenset of extra params for file lookup)
GRID_ROWS = [
    ("manual",              "manual",         None),
    ("optix",               "optix",          None),
    ("adaptive",            "adaptive",       {}),
    ("bricked-regions\nbrick16 am-off", "bricked-regions", {"brick": "16", "am": "off"}),
    ("octree\nleaf8 am-off",            "octree",          {"leaf": "8",  "am": "off"}),
    ("octree-regions\nleaf8 am-off",    "octree-regions",  {"leaf": "8",  "am": "off"}),
    ("nanovdb\nnearest",                "nanovdb",         {"sampler": "nearest"}),
    ("nanovdb\ntrilinear",              "nanovdb",         {"sampler": "trilinear"}),
]

# ── PPM filename parsing ─────────────────────────────────────
def parse_ppm(basename):
    if basename in ("manual", "optix"):
        return basename, None, {}
    m = re.match(r"^(.+)--(eps\d{3})((?:-[^-]+)*)$", basename)
    if not m:
        return basename, None, {}
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


# ── foreground mask ──────────────────────────────────────────
def foreground_mask(ref, threshold=5):
    """Binary mask: True where ANY channel > threshold (non-background)."""
    return np.any(ref > threshold, axis=2)


# ── image quality metrics (foreground-masked) ────────────────
def _gray(a):
    return 0.299 * a[..., 0].astype(np.float64) + \
           0.587 * a[..., 1].astype(np.float64) + \
           0.114 * a[..., 2].astype(np.float64)


def compute_metrics(img, ref, mask):
    """Return dict with psnr, ssim, mae, pct_changed for foreground region."""
    fimg = img[mask].astype(np.float64)
    fref = ref[mask].astype(np.float64)

    # MAE (per-pixel mean absolute error, 0-255 scale)
    mae = np.mean(np.abs(fimg - fref))

    # PSNR (from MSE across all 3 channels)
    mse = np.mean((fimg - fref) ** 2)
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10.0 * np.log10(255.0 ** 2 / mse)

    # Percentage of changed pixels (any channel differs by > 2)
    diff = np.abs(fimg - fref)
    changed = np.any(diff > 2.0, axis=1)
    pct_changed = 100.0 * changed.sum() / max(changed.size, 1)

    # SSIM (luminance, windowed via Gaussian weighting)
    lum_img = _gray(img)
    lum_ref = _gray(ref)
    win = 7
    sig = 1.5
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_img = gaussian_filter(lum_img, sigma=sig)
    mu_ref = gaussian_filter(lum_ref, sigma=sig)
    mu_img2 = mu_img ** 2
    mu_ref2 = mu_ref ** 2
    mu_cross = mu_img * mu_ref

    sig_img2 = gaussian_filter(lum_img ** 2, sigma=sig) - mu_img2
    sig_ref2 = gaussian_filter(lum_ref ** 2, sigma=sig) - mu_ref2
    sig_cross = gaussian_filter(lum_img * lum_ref, sigma=sig) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sig_cross + C2)) / \
               ((mu_img2 + mu_ref2 + C1) * (sig_img2 + sig_ref2 + C2))

    # only consider foreground pixels in SSIM map
    ssim_val = np.mean(ssim_map[mask])

    return {
        "psnr": psnr,
        "ssim": ssim_val,
        "mae": mae,
        "pct_changed": pct_changed,
    }


# ── color for metric value ───────────────────────────────────
def metric_color(metric_name, value):
    """Return a color string based on quality thresholds."""
    if metric_name == "psnr":
        if value >= 40: return "#00ff00"   # excellent
        if value >= 30: return "#aaff00"   # good
        if value >= 25: return "#ffaa00"   # acceptable
        return "#ff4444"                    # poor
    if metric_name == "ssim":
        if value >= 0.98: return "#00ff00"
        if value >= 0.95: return "#aaff00"
        if value >= 0.90: return "#ffaa00"
        return "#ff4444"
    if metric_name == "mae":
        if value <= 2:  return "#00ff00"
        if value <= 5:  return "#aaff00"
        if value <= 10: return "#ffaa00"
        return "#ff4444"
    if metric_name == "pct_changed":
        if value <= 2:  return "#00ff00"
        if value <= 5:  return "#aaff00"
        if value <= 15: return "#ffaa00"
        return "#ff4444"
    return "white"


# ── main ─────────────────────────────────────────────────────
def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    ds_dir = RESULTS_ROOT / dataset if dataset else None
    if ds_dir is None or not ds_dir.is_dir():
        available = sorted(d.name for d in RESULTS_ROOT.iterdir() if d.is_dir())
        sys.exit(
            f"Dataset '{dataset}' not found.\nAvailable:\n"
            + "\n".join(f"  {a}" for a in available[:10])
            + ("\n  ..." if len(available) > 10 else "")
        )

    # scan PPMs
    ppm_files = {}
    for f in ds_dir.glob("*.ppm"):
        mode, eps_code, params = parse_ppm(f.stem)
        ppm_files[(mode, eps_code, frozenset(params.items()))] = str(f)

    # manual reference
    manual_key = ("manual", None, frozenset())
    if manual_key not in ppm_files:
        sys.exit(f"No manual snapshot in {ds_dir}")
    man_img = np.asarray(Image.open(ppm_files[manual_key]), dtype=np.uint8)
    mask = foreground_mask(man_img)

    # ── build grid: (row_idx, eps_code) -> ppm_path ──────────
    grid = {}
    for row_idx, (_label, mode, variant) in enumerate(GRID_ROWS):
        if variant is None:
            # manual / optix: single snapshot, show at eps=0.01 only
            key = (mode, None, frozenset())
            if key in ppm_files:
                grid[(row_idx, "eps001")] = ppm_files[key]
        else:
            for eps_code in EPS_CODES:
                key = (mode, eps_code, frozenset(variant.items()))
                if key in ppm_files:
                    grid[(row_idx, eps_code)] = ppm_files[key]

    n_rows = len(GRID_ROWS)
    n_eps = len(EPSILONS)

    # ── compute per-cell metrics once (shared by grid + table) ─
    metrics = {}
    for key, path in grid.items():
        img = np.asarray(Image.open(path), dtype=np.uint8)
        metrics[key] = compute_metrics(img, man_img, mask)

    # ── figure 1: image grid ─────────────────────────────────
    # Dedicated header row for the epsilon labels (guarantees a
    # shared baseline) + near-zero vertical spacing between cells.
    im = np.asarray(Image.open(next(iter(grid.values()))), dtype=np.uint8)
    aspect = im.shape[1] / im.shape[0]           # image aspect (W/H)
    CELL_W = 2.4                                  # inches per image column
    LABEL_W = 1.4                                 # inches for strategy column
    H_CELL = CELL_W / aspect                      # cell height for the image
    HEADER_H = 0.45                               # header row height (units)

    fig_grid = plt.figure(figsize=(
        LABEL_W + n_eps * CELL_W,
        0.10 + (HEADER_H + n_rows) * H_CELL + 0.6,  # top 10% is the suptitle
    ))
    gs_img = fig_grid.add_gridspec(
        n_rows + 1, n_eps + 1,
        height_ratios=[HEADER_H] + [1.0] * n_rows,
        width_ratios=[LABEL_W] + [CELL_W] * n_eps,
        hspace=0.2, wspace=0.0,
        top=0.90, bottom=0.02, left=0.01, right=0.99,
    )

    for row_idx, (label, _mode, _variant) in enumerate(GRID_ROWS):
        ax = fig_grid.add_subplot(gs_img[row_idx + 1, 0])
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=7.5, fontfamily="monospace")
        ax.axis("off")

        for col_idx, eps_code in enumerate(EPS_CODES):
            ax = fig_grid.add_subplot(gs_img[row_idx + 1, col_idx + 1])
            if (row_idx, eps_code) in grid:
                img = np.asarray(Image.open(grid[(row_idx, eps_code)]), dtype=np.uint8)
                m = metrics[(row_idx, eps_code)]
                ax.imshow(img, interpolation="nearest")
                txt = (f"PSNR {m['psnr']:.1f} dB\n"
                       f"SSIM {m['ssim']:.4f}\n"
                       f"MAE  {m['mae']:.2f}\n"
                       f"Chg  {m['pct_changed']:.1f}%")
                ax.text(0.02, 0.98, txt,
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=5.8, color="white", linespacing=1.25,
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="black",
                                  ec="none", alpha=0.50))
            ax.axis("off")

    # header row: shared baseline for column titles + strategy label
    hdr = fig_grid.add_subplot(gs_img[0, 0])
    hdr.text(0.5, 0.5, "strategy", ha="center", va="center",
             fontsize=8, fontweight="bold")
    hdr.axis("off")
    for col_idx, eps_code in enumerate(EPS_CODES):
        hdr = fig_grid.add_subplot(gs_img[0, col_idx + 1])
        hdr.text(0.5, 0.5, f"eps = {EPSILONS[col_idx]}",
                 ha="center", va="center", fontsize=8, fontweight="bold")
        hdr.axis("off")

    # fig_grid.suptitle(f"Strategy comparison — {dataset}\n"
    #                   f"(foreground-masked metrics vs manual reference)",
    #                   fontsize=12, y=0.96)
    out_grid = PLOT_DIR / f"d10_strategy_grid_{dataset}_grid.png"
    fig_grid.savefig(out_grid, dpi=150)
    plt.close(fig_grid)
    print(f"  saved {out_grid}")

    # ── figure 2: metrics summary table ──────────────────────
    fig_tbl = plt.figure(figsize=(3.4 * (n_eps + 2), 1.0 * n_rows + 1.8))
    ax_tbl = fig_tbl.add_subplot(111)
    ax_tbl.axis("off")

    col_labels = [f"eps={e}" for e in EPSILONS]
    row_labels = []
    cell_text = []
    cell_colors = []
    for row_idx, (label, _mode, _variant) in enumerate(GRID_ROWS):
        row_labels.append(label.replace("\n", " "))
        row_text = []
        row_col = []
        for col_idx, eps_code in enumerate(EPS_CODES):
            if (row_idx, eps_code) in grid:
                m = metrics[(row_idx, eps_code)]
                row_text.append(
                    f"PSNR={m['psnr']:.1f}  SSIM={m['ssim']:.4f}\n"
                    f"MAE={m['mae']:.2f}   Chg={m['pct_changed']:.1f}%"
                )
                row_col.append(metric_color("psnr", m["psnr"]))
            else:
                row_text.append("—")
                row_col.append("#333333")
        cell_text.append(row_text)
        cell_colors.append(row_col)

    tbl = ax_tbl.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.5)
    tbl.scale(1.0, 1.6)

    # style header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # style row labels
    for i in range(len(row_labels)):
        tbl[i + 1, -1].set_facecolor("#D9E2F3")
        tbl[i + 1, -1].set_text_props(fontweight="bold", fontsize=5)
        for j in range(len(EPSILONS)):
            tbl[i + 1, j].set_facecolor(cell_colors[i][j])
            tbl[i + 1, j].set_text_props(color="black" if cell_colors[i][j] in ("#00ff00", "#aaff00") else "black")

    # fig_tbl.suptitle(f"Foreground-masked fidelity metrics — {dataset}", fontsize=12)

    # legend text
    ax_tbl.text(
        0.5, -0.05,
        "color key:  PSNR>=40=green  >=30=lime  >=25=orange  <25=red   "
        "|  foreground threshold: ref intensity > 5",
        transform=ax_tbl.transAxes, ha="center", va="top",
        fontsize=5.5, color="#888888",
    )

    out_tbl = PLOT_DIR / f"d10_strategy_grid_{dataset}_table.png"
    fig_tbl.savefig(out_tbl, dpi=150)
    plt.close(fig_tbl)
    print(f"  saved {out_tbl}")


if __name__ == "__main__":
    main()
