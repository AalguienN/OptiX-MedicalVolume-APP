# ============================================================
# Shared helpers for all analysis scripts.
#   load()   -> DataFrame with the tuple-"," fix + GB column
#   PLOT_DIR -> common output folder for every PNG
# ============================================================
import re
import io
import os

import pandas as pd

CSV_PATH = "/home/adri/projects/TFM/optix_test_results/results.csv"
PLOT_DIR = "/home/adri/projects/TFM/optix_clone/python_visualization_tests/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load():
    raw = open(CSV_PATH).read()
    fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
    df = pd.read_csv(io.StringIO(fixed))
    df["gpu_mem_gb"] = df["gpu_mem_bytes"] / 1e9
    return df


def save(fig, name):
    out = os.path.join(PLOT_DIR, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")
    return out


def iqr_outliers(s):
    """Boolean mask of outliers via the classic 1.5*IQR rule."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def clip_col(s, lo=0.01, hi=0.99):
    """Winsorize a series at the given quantiles (outlier-robust clip)."""
    return s.clip(s.quantile(lo), s.quantile(hi))