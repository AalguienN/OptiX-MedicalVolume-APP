# ============================================================
# Populate the Rendering performance table (subsec:ResultsPerformance)
# from optix_test_results2/results.csv (47 datasets x 86 configs).
#
# Methodology (matches Section \ref{subsec:ResultsPerformance}):
#   * per-dataset cell render  = aggregate of the strategy's variants
#     at that epsilon (median by default) for that dataset
#   * per-dataset speedup      = manual_render_mean_ms / cell_render
#   * cell summary             = median over the test set (datasets)
#
# results.csv stores the spacing/origin tuples with unquoted commas,
# so it is pre-fixed with a regex that quotes those fields (same trick
# as viz_common.load / d11).
#
# Run:
#   venv/bin/python rendering_performance_table.py
#   venv/bin/python rendering_performance_table.py --variant-agg mean
#   venv/bin/python rendering_performance_table.py --results-dir /path --save
# ============================================================
import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("/home/adri/projects/TFM/optix_test_results2")

EPSILONS = [0.01, 0.05, 0.10, 0.20]
MODE_ORDER = ["manual", "optix", "adaptive", "bricked-regions",
              "octree", "octree-regions", "nanovdb"]
BASELINES = {"manual", "optix"}


def load(csv_path: Path) -> pd.DataFrame:
    raw = open(csv_path).read()
    fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
    return pd.read_csv(io.StringIO(fixed))


def cell_summary(g: pd.DataFrame, agg: str) -> dict:
    """Aggregate a (dataset, mode, epsilon) group's variants into one cell."""
    d = agg if isinstance(agg, dict) else agg
    return {
        "render": g["render_mean_ms"].aggregate(agg),
        "fps": g["fps_mean"].aggregate(agg),
        "skip": g["skip_ratio_bounds"].aggregate(agg),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the rendering-performance "
                                            "table for subsec:ResultsPerformance")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--variant-agg", choices=["median", "mean", "min"],
                    default="median",
                    help="how to collapse a strategy's variants (brick/leaf/"
                         "amode/sampler) inside one (strategy, epsilon, dataset) "
                         "cell before the per-dataset speedup (default median)")
    ap.add_argument("--save", action="store_true",
                    help="also write the LaTeX snippet to plots/")
    args = ap.parse_args()

    csv_path = args.results_dir / "results.csv"
    df = load(csv_path)
    print(f"{len(df)} rows loaded across {df['dataset'].nunique()} datasets")

    # ---- collapse variants into per-dataset cells ----
    cells = df.groupby(["dataset", "mode", "epsilon"], as_index=False).agg(
        render=("render_mean_ms", args.variant_agg),
        fps=("fps_mean", args.variant_agg),
        skip=("skip_ratio_bounds", args.variant_agg),
    )

    # ---- per-dataset speedup vs the manual baseline ----
    manual = cells[cells["mode"] == "manual"].set_index("dataset")["render"]
    cells["speedup"] = cells["dataset"].map(manual) / cells["render"]

    # ---- median over the test set per (mode, epsilon) ----
    summary = cells.groupby(["mode", "epsilon"], as_index=False).agg(
        speedup=("speedup", "median"),
        render=("render", "median"),
        fps=("fps", "median"),
        skip=("skip", "median"),
    )
    idx = summary.set_index(["mode", "epsilon"])

    # ---- console preview ----
    print("\nMedian over the test set (variant-agg=%s):" % args.variant_agg)
    print(f"{'strategy':<16}{'eps':>6}{'speedup':>9}{'render_ms':>11}"
          f"{'fps':>9}{'skip':>9}")
    for mode in MODE_ORDER:
        if mode in BASELINES:
            r = idx.loc[(mode, df["epsilon"].iloc[0])]
            print(f"{mode:<16}{'---':>6}{r['speedup']:>9.3f}{r['render']:>11.2f}"
                  f"{r['fps']:>9.1f}{r['skip']:>9.3f}")
        else:
            for e in EPSILONS:
                r = idx.loc[(mode, e)]
                print(f"{mode:<16}{e:>6.2f}{r['speedup']:>9.3f}"
                      f"{r['render']:>11.2f}{r['fps']:>9.1f}{r['skip']:>9.3f}")

    # ---- inform the "Several patterns emerge" bullet ----
    print("\nPattern hints (median speedup per cell, variant-agg=%s):" %
          args.variant_agg)
    per_eps = summary.set_index("epsilon")
    for e in EPSILONS:
        order = (idx.loc[[(m, e) for m in MODE_ORDER if m not in BASELINES]]
                 .reset_index().sort_values("speedup", ascending=False))
        best = order.iloc[0]
        print(f"  eps {e:.2f}: fastest {best['mode']} "
              f"(speedup {best['speedup']:.2f})")
    for mode in MODE_ORDER:
        if mode in BASELINES:
            continue
        sub = idx.loc[[(mode, e) for e in EPSILONS]].reset_index()
        best = sub.loc[sub["speedup"].idxmax()]
        print(f"  {mode}: best at eps {best['epsilon']:.2f} "
              f"(speedup {best['speedup']:.2f})")

    # ---- LaTeX rows matching table tab:perf_per_eps ----
    s_fmt, r_fmt, f_fmt, k_fmt = ("{:.2f}", "{:.2f}", "{:.0f}", "{:.3f}")
    out = []
    out.append(r"\begin{tabular}{l c r r r r}")
    out.append(r"    \textbf{Strategy} & $\epsilon$ & \textbf{Speedup} & "
               r"\textbf{render\_mean\_ms} & \textbf{FPS} & \textbf{Skip ratio} \\")
    out.append(r"    \midrule")
    out.append(r"    \addlinespace[0.4em]")
    for mode in MODE_ORDER:
        if mode in BASELINES:
            r = idx.loc[(mode, df["epsilon"].iloc[0])]
            out.append(r"    \multicolumn{1}{l}{\texttt{%s}} & --- & %s & %s & %s & %s \\"
                       % (mode, s_fmt.format(r["speedup"]), r_fmt.format(r["render"]),
                          f_fmt.format(r["fps"]), k_fmt.format(r["skip"])))
        else:
            out.append(r"    \addlinespace[0.4em]")
            out.append(r"    \multicolumn{6}{l}{\texttt{%s}} \\" % mode)
            for e in EPSILONS:
                r = idx.loc[(mode, e)]
                out.append(r"    & %.2f & %s & %s & %s & %s \\"
                           % (e, s_fmt.format(r["speedup"]), r_fmt.format(r["render"]),
                              f_fmt.format(r["fps"]), k_fmt.format(r["skip"])))
    out.append(r"    \bottomrule")
    out.append(r"\end{tabular}")

    print("\n" + "=" * 78)
    print("LaTeX rows for tab:perf_per_eps "
          "(median, variant-agg=%s)" % args.variant_agg)
    print("=" * 78)
    print("\n".join(out))

    if args.save:
        plot_dir = Path(__file__).resolve().parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        tex = plot_dir / "perf_per_eps_table.tex"
        tex.write_text("\n".join(out99) + "\n")
        print(f"\n  saved {tex}")


if __name__ == "__main__":
    main()