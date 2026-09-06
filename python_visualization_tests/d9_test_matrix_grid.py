# ============================================================
# D9 — Full test-matrix image grid (run_tests2.sh)
#   Visual validation figure: every configuration from the
#   cross-product sweep rendered as a grid for one dataset.
#   Rows: manual, optix, then all param-variants per mode
#   Columns: epsilon (0.01, 0.05, 0.1, 0.2)
#   Each cell: rendered image + Δ-lum / Δ-RGB vs manual ref.
#
#   Usage: venv/bin/python d9_test_matrix_grid.py [dataset]
# ============================================================
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "optix_test_results2"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

EPSILONS = [0.01, 0.05, 0.1, 0.2]
EPS_CODES = ["eps001", "eps005", "eps010", "eps020"]

# ── grid definition ──────────────────────────────────────────
# (row_label, mode, variant_key_or_None)
# variant_key: dict identifying the sub-params, matched against parsed PPM info
# None means the config only appears at a single epsilon (manual, optix)
GRID_ROWS = [
    ("manual",                      "manual",     None),
    ("optix",                       "optix",      None),
    ("adaptive",                    "adaptive",   {}),
    ("bricked-regions | brick8  | am-off",  "bricked-regions", {"brick": "8",  "am": "off"}),
    ("bricked-regions | brick8  | am-on",   "bricked-regions", {"brick": "8",  "am": "on"}),
    ("bricked-regions | brick16 | am-off",  "bricked-regions", {"brick": "16", "am": "off"}),
    ("bricked-regions | brick16 | am-on",   "bricked-regions", {"brick": "16", "am": "on"}),
    ("bricked-regions | brick32 | am-off",  "bricked-regions", {"brick": "32", "am": "off"}),
    ("bricked-regions | brick32 | am-on",   "bricked-regions", {"brick": "32", "am": "on"}),
    ("octree | leaf4  | am-off",            "octree",          {"leaf": "4",  "am": "off"}),
    ("octree | leaf4  | am-on",             "octree",          {"leaf": "4",  "am": "on"}),
    ("octree | leaf8  | am-off",            "octree",          {"leaf": "8",  "am": "off"}),
    ("octree | leaf8  | am-on",             "octree",          {"leaf": "8",  "am": "on"}),
    ("octree | leaf16 | am-off",            "octree",          {"leaf": "16", "am": "off"}),
    ("octree | leaf16 | am-on",             "octree",          {"leaf": "16", "am": "on"}),
    ("octree-regions | leaf4  | am-off",    "octree-regions",  {"leaf": "4",  "am": "off"}),
    ("octree-regions | leaf4  | am-on",     "octree-regions",  {"leaf": "4",  "am": "on"}),
    ("octree-regions | leaf8  | am-off",    "octree-regions",  {"leaf": "8",  "am": "off"}),
    ("octree-regions | leaf8  | am-on",     "octree-regions",  {"leaf": "8",  "am": "on"}),
    ("octree-regions | leaf16 | am-off",    "octree-regions",  {"leaf": "16", "am": "off"}),
    ("octree-regions | leaf16 | am-on",     "octree-regions",  {"leaf": "16", "am": "on"}),
    ("nanovdb | nearest",                   "nanovdb",         {"sampler": "nearest"}),
    ("nanovdb | trilinear",                 "nanovdb",         {"sampler": "trilinear"}),
]


# ── PPM filename parsing ─────────────────────────────────────
def parse_ppm(basename):
    """Parse a PPM basename (no extension) into (mode, epsilon_code, params).

    Returns (mode, eps_code, params_dict).
    Examples:
      'manual'                         -> ('manual', None, {})
      'optix'                          -> ('optix',  None, {})
      'adaptive--eps001'               -> ('adaptive', 'eps001', {})
      'bricked-regions--eps005-brick16-amoff'
                                       -> ('bricked-regions', 'eps005', {'brick':'16','am':'off'})
      'nanovdb--eps010-trilinear'      -> ('nanovdb', 'eps010', {'sampler':'trilinear'})
    """
    if basename in ("manual", "optix"):
        return basename, None, {}

    m = re.match(
        r"^(.+)--(eps\d{3})((?:-[^-]+)*)$",
        basename,
    )
    if not m:
        return basename, None, {}

    mode = m.group(1)
    eps_code = m.group(2)
    tail = m.group(3).lstrip("-")

    params = {}
    if tail:
        for part in tail.split("-"):
            if part.startswith("brick"):
                params["brick"] = part[5:]
            elif part.startswith("leaf"):
                params["leaf"] = part[4:]
            elif part.startswith("am"):
                params["am"] = part[2:]
            elif part in ("nearest", "trilinear"):
                params["sampler"] = part
    return mode, eps_code, params


# ── difference metrics ───────────────────────────────────────
def _gray(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def diff_pcts(a, b):
    lum = 100.0 * np.mean(np.abs(_gray(a) - _gray(b))) / 255.0
    rgb = 100.0 * np.mean(
        np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=2)
    ) / 255.0
    return lum, rgb


# ── main ─────────────────────────────────────────────────────
def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    ds_dir = RESULTS_ROOT / dataset if dataset else None
    if ds_dir is None or not ds_dir.is_dir():
        available = sorted(
            d.name for d in RESULTS_ROOT.iterdir() if d.is_dir()
        )
        sys.exit(
            f"Dataset '{dataset}' not found.\nAvailable:\n"
            + "\n".join(f"  {a}" for a in available[:10])
            + ("\n  ..." if len(available) > 10 else "")
        )

    # scan PPM files
    ppm_files = {}
    for f in ds_dir.glob("*.ppm"):
        mode, eps_code, params = parse_ppm(f.stem)
        ppm_files[(mode, eps_code, frozenset(params.items()))] = str(f)

    # find manual reference
    manual_key = ("manual", None, frozenset())
    if manual_key not in ppm_files:
        sys.exit(f"No manual snapshot in {ds_dir}")
    man_img = np.asarray(Image.open(ppm_files[manual_key]), dtype=np.uint8)

    # build grid mapping: (row_idx, eps_code) -> ppm_path
    grid = {}
    for row_idx, (label, mode, variant) in enumerate(GRID_ROWS):
        if variant is None:
            # manual / optix: only one snapshot, show in eps001 column
            key = (mode, None, frozenset())
            if key in ppm_files:
                grid[(row_idx, "eps001")] = ppm_files[key]
        else:
            for eps_code in EPS_CODES:
                key = (mode, eps_code, frozenset(variant.items()))
                if key in ppm_files:
                    grid[(row_idx, eps_code)] = ppm_files[key]

    n_rows = len(GRID_ROWS)
    n_cols = len(EPSILONS)
    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=(3.4 * (n_cols + 1), 1.7 * n_rows + 1.0),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (label, _mode, _variant) in enumerate(GRID_ROWS):
        # label column
        ax_label = axes[row_idx, 0]
        ax_label.text(
            0.5, 0.5, label, ha="center", va="center",
            fontsize=7, fontfamily="monospace", wrap=True,
        )
        ax_label.axis("off")
        if row_idx == 0:
            ax_label.set_title("strategy", fontsize=9)

        # epsilon columns
        for col_idx, eps_code in enumerate(EPS_CODES):
            ax = axes[row_idx, col_idx + 1]
            if (row_idx, eps_code) in grid:
                img = np.asarray(
                    Image.open(grid[(row_idx, eps_code)]), dtype=np.uint8
                )
                ax.imshow(img, interpolation="nearest")
                lum_pct, rgb_pct = diff_pcts(img, man_img)
                color = "lime" if lum_pct < 2.0 else ("yellow" if lum_pct < 5.0 else "red")
                ax.text(
                    0.02, 0.98,
                    f"\u0394 lum {lum_pct:.1f}%\n\u0394 rgb {rgb_pct:.1f}%",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color="white", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.25", fc="black",
                              ec="none", alpha=0.45),
                )
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"eps = {EPSILONS[col_idx]}", fontsize=9)

    fig.suptitle(
        f"Full test matrix snapshots — {dataset}", fontsize=13,
    )
    fig.tight_layout()
    out = PLOT_DIR / f"d9_test_matrix_grid_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
