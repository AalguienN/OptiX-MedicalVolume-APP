# ============================================================
# Populate the Latency results section (sec:ResultsLatency)
# from optix_test_results3 (interactive campaign with real input).
#
# Method (matches sec:ResultsLatency):
#   * interactive runs record a latency sample per input event:
#       latency = next-finalized-frame timestamp - input-event timestamp
#   * "values not set to -1": frames where no input event arrived
#     carry the -1.000 sentinel and are dropped
#   * per (mode) the latency samples of ALL variants/epsilons/runs
#     of the interactive campaign are pooled (long format) and
#     summarized by median / p90 / p95 / p99 / max and the fraction
#     above the 50 ms responsiveness budget (sec:DatasetRequirements)
#   * a run-level view (median latency of each run, then median
#     across runs) is printed as a cross-check
#
# The on-disk per-frame CSV of a run is <dataset>/<strategy>.csv, where
# the strategy label uses '--' where the results.csv 'strategy' column
# uses '_' (e.g. adaptive--eps001.csv <-> adaptive_eps001).
# results.csv stores the spacing/origin tuples with unquoted commas,
# so it is pre-fixed with a regex that quotes those fields (same trick
# as viz_common.load / rendering_performance_table.py).
#
# Run:
#   venv/bin/python latency_table.py
#   venv/bin/python latency_table.py --results-dir /path --save
# ============================================================
import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("/home/adri/projects/TFM/optix_test_results3")
BUDGET_MS = 50.0
MODE_ORDER = ["manual", "optix", "adaptive", "bricked-regions",
              "octree", "octree-regions", "nanovdb"]


def load(csv_path: Path) -> pd.DataFrame:
    raw = open(csv_path).read()
    fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
    return pd.read_csv(io.StringIO(fixed))


def collect_latency(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Return one row per non-sentinel latency sample (long format)."""
    rows = []

    def data_csv(meta: pd.Series) -> Path | None:
        stem = meta["dataset"]
        for fname in (f"{stem}/{meta['strategy'].replace('_', '--')}.csv",
                      f"{stem}/{meta['strategy']}.csv"):
            p = run_dir / fname
            if p.exists():
                return p
        return None

    for _, meta in df.iterrows():
        csv_path = data_csv(meta)
        if csv_path is None:
            continue
        lat = pd.read_csv(csv_path)["latency_ms"]
        lat = lat[lat >= 0]
        if lat.empty:
            continue
        for v in lat:
            rows.append(dict(dataset=meta["dataset"], mode=meta["mode"],
                             epsilon=meta["epsilon"], strategy=meta["strategy"],
                             latency_ms=v))
    return pd.DataFrame(rows)


def summarize(g: pd.DataFrame, budget_ms: float = BUDGET_MS) -> dict:
    v = g["latency_ms"]
    return {
        "n": int(v.size),
        "median": float(v.median()),
        "mean": float(v.mean()),
        "p90": float(v.quantile(0.90)),
        "p95": float(v.quantile(0.95)),
        "p99": float(v.quantile(0.99)),
        "max": float(v.max()),
        "over_budget": float((v > budget_ms).mean() * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the latency table "
                                            "for sec:ResultsLatency")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--budget-ms", type=float, default=BUDGET_MS,
                    help="responsiveness budget (default 50 ms)")
    ap.add_argument("--save", action="store_true",
                    help="also write the LaTeX snippet to plots/")
    args = ap.parse_args()
    budget_ms = args.budget_ms

    df = load(args.results_dir / "results.csv")
    print(f"{len(df)} runs loaded across {df['dataset'].nunique()} datasets")

    long = collect_latency(df, args.results_dir)
    nonneg = df[df["latency_latest_ms"] >= 0]
    print(f"runs with interactive input (summary col): {len(nonneg)} of {len(df)}; "
          f"runs matched with per-frame latency: "
          f"{long[['dataset','strategy']].drop_duplicates().shape[0]}; "
          f"latency samples pooled: {long.shape[0]}")
    if long.empty:
        print("no latency samples found; nothing to report")
        return

    # ---- run-level view (cross-check; per-run median then median) ----
    runlev = (long.groupby(["dataset", "strategy"])["latency_ms"].median()
              .reset_index().merge(df[["dataset", "strategy", "mode"]],
                                   on=["dataset", "strategy"])
              .groupby("mode")["latency_ms"].agg(["median", "mean", "max"]))
    print("\nRun-level cross-check: median of per-run median latency (ms):")
    for m in MODE_ORDER:
        if m in runlev.index:
            print(f"  {m:<16} {runlev.loc[m, 'median']:7.2f}  "
                  f"(worst-run median {runlev.loc[m, 'max']:6.2f})")

    # ---- pooled (event-level) stats per strategy ----
    stats = long.groupby("mode", as_index=False).apply(
        lambda g: pd.Series(summarize(g)), include_groups=False)
    print("\nPooled event-level latency per strategy:")
    print(f"{'strategy':<16}{'events':>8}{'p50':>7}{'mean':>7}{'p90':>7}"
          f"{'p95':>7}{'p99':>7}{'max':>7}{'%>budget':>9}")
    for _, r in stats.iterrows():
        print(f"{r['mode']:<16}{r['n']:>8}{r['median']:>7.2f}{r['mean']:>7.2f}"
              f"{r['p90']:>7.2f}{r['p95']:>7.2f}{r['p99']:>7.2f}"
              f"{r['max']:>7.2f}{r['over_budget']:>9.1f}")

    # ---- per-strategy, per-epsilon pooled medians (context) ----
    pe = long.groupby(["mode", "epsilon"], as_index=False).apply(
        lambda g: pd.Series(summarize(g)), include_groups=False)
    idx = pe.set_index(["mode", "epsilon"])
    print("\nPooled median latency by (mode, eps):")
    for m in MODE_ORDER:
        if m in ("manual", "optix"):
            sub = long[long["mode"] == m]
            print(f"  {m:<16} eps ---  p50 {sub['latency_ms'].median():6.2f} "
                  f"(n={sub.shape[0]})")
        else:
            vals = "  ".join(f"{e:.2f}:{idx.loc[(m, e), 'median']:.2f}"
                             for e in [0.01, 0.05, 0.10, 0.20])
            print(f"  {m:<16} {vals}")

    # ---- LaTeX table for tab:latency ----
    st = stats.set_index("mode")
    out = []
    out.append(r"\begin{tabular}{l r r r r r r}")
    out.append(r"    \textbf{Strategy} & \textbf{\# events} & \textbf{median} & "
               r"\textbf{p90} & \textbf{p95} & \textbf{p99} & \textbf{max} \\")
    out.append(r"        & & (ms) & (ms) & (ms) & (ms) & (ms) \\")
    out.append(r"    \midrule")
    for m in MODE_ORDER:
        if m not in st.index:
            continue
        r = st.loc[m]
        out.append(r"    \texttt{%-10s} & %d & %.2f & %.2f & %.2f & %.2f & %.2f \\"
                   % (m, int(r["n"]), r["median"], r["p90"], r["p95"],
                      r["p99"], r["max"]))
    out.append(r"    \bottomrule")
    out.append(r"\end{tabular}")

    print("\n" + "=" * 78)
    print("LaTeX rows for tab:latency (50 ms budget)")
    print("=" * 78)
    print("\n".join(out))

    if args.save:
        plot_dir = Path(__file__).resolve().parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        tex = plot_dir / "latency_table.tex"
        tex.write_text("\n".join(out) + "\n")
        print(f"\n  saved {tex}")


if __name__ == "__main__":
    main()