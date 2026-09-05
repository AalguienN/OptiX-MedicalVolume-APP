# ============================================================
# D1 — Image fidelity vs the manual reference.
#   For every run: load snapshot + its dataset's manual.ppm and
#   compute fidelity metrics in BOTH colour modes:
#     _gs  = grayscale luminance (0.299R + 0.587G + 0.114B)
#     _rgb = per-channel R, G, B metrics, averaged
#   Metrics: PSNR, SSIM, MAE, max-abs-diff, foreground-masked
#   PSNR (luma > 5), and the % of pixels that differ by > 10.
#   Sanity check: optix renders the same full pose as manual so
#   it must come out pixel-identical (PSNR = inf, SSIM = 1).
#   Output: /home/adri/projects/TFM/optix_test_results/
#           image_fidelity_metrics.csv
# ============================================================
import os

import viz_common as vc
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

OUT_CSV = "/home/adri/projects/TFM/optix_test_results/image_fidelity_metrics.csv"
FG_THRESH = 5.0        # luminance above this counts as "content", not background
DIFF_THRESH = 10.0     # |Δ| above this counts as a visibly-changed pixel

_CACHE = {}


def progress(i, n, width=50):
    """Tiny dependency-free progress bar."""
    pct = (i + 1) / n
    filled = int(width * pct)
    bar = "=" * filled + " " * (width - filled)
    print(f"\r[{bar}] {100 * pct:5.1f}% ({i + 1}/{n})  ", end="", flush=True)


def _gray(a):
    """Grayscale luminance of an RGB float array."""
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _load(path):
    """Load an image as an RGB float array, cached by path."""
    if path not in _CACHE:
        _CACHE[path] = np.asarray(Image.open(path), dtype=np.float64)
    return _CACHE[path]


def _ssim(x, y, w=11, k1=0.01, k2=0.03):
    """Grayscale SSIM with a Gaussian window (scipy), classic params."""
    C1, C2 = (k1 * 255) ** 2, (k2 * 255) ** 2
    mu_x = gaussian_filter(x, w / 5, truncate=1.0)
    mu_y = gaussian_filter(y, w / 5, truncate=1.0)
    sig_xy = gaussian_filter(x * y, w / 5, truncate=1.0) - mu_x * mu_y
    sig_x2 = gaussian_filter(x * x, w / 5, truncate=1.0) - mu_x * mu_x
    sig_y2 = gaussian_filter(y * y, w / 5, truncate=1.0) - mu_y * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2)
    return np.mean(num / den)


def _metrics(a, b):
    """Fidelity of the single-channel arrays b vs reference a."""
    d = a - b
    mse = np.mean(d ** 2)
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
    mae = float(np.mean(np.abs(d)))
    maxd = float(np.max(np.abs(d)))
    mask = a > FG_THRESH               # foreground-masked metrics
    mse_m = np.mean(d[mask] ** 2) if mask.any() else np.nan
    psnr_m = float("inf") if (mask.any() and mse_m == 0) \
        else (np.nan if not mask.any() else 20 * np.log10(255.0) - 10 * np.log10(mse_m))
    pct_changed = 100.0 * np.mean(np.abs(d) > DIFF_THRESH)
    return psnr, _ssim(a, b), mae, maxd, psnr_m, pct_changed


def main():
    df = vc.load()
    out_rows = []

    for i, r in df.iterrows():
        # reference = the manual snapshot of the same dataset
        manual_rows = df[(df["dataset"] == r["dataset"]) & (df["mode"] == "manual")]
        man = _load(manual_rows["snapshot_path"].iloc[0])
        img = _load(r["snapshot_path"])

        row = {"dataset": r["dataset"], "strategy": r["strategy"], "mode": r["mode"],
               "epsilon": r["epsilon"]}

        # grayscale
        g_man, g_img = _gray(man), _gray(img)
        g = _metrics(g_man, g_img)
        row.update(dict(zip(
            ["psnr_gs", "ssim_gs", "mae_gs", "maxdiff_gs", "psnr_masked_gs", "pct_changed_gs"],
            g)))

        # per-channel RGB, averaged
        per = np.mean([_metrics(man[..., c], img[..., c]) for c in range(3)], axis=0)
        row.update(dict(zip(
            ["psnr_rgb", "ssim_rgb", "mae_rgb", "maxdiff_rgb", "psnr_masked_rgb", "pct_changed_rgb"],
            per)))

        out_rows.append(row)
        progress(i, len(df))

    print()
    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV}")

    # sanity: optix must be identical to manual
    opt = out[out["mode"] == "optix"]
    print(f"Sanity optix: {len(opt)} rows, "
          f"PSNR>1e9: {(opt['psnr_gs'].fillna(0) > 1e9).all()}, "
          f"SSIM==1: {np.isclose(opt['ssim_gs'], 1.0).all()}")


if __name__ == "__main__":
    main()