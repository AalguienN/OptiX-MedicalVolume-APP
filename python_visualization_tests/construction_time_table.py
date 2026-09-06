# ============================================================
# Populate the Construction time table (subsec:ResultsConstruction)
# from the measurement logs in optix_test_results3.
#
# Data source: per-strategy *.log files. The authoritative value
# is the "Construction time: X.XXX ms" line. results.csv cannot be
# parsed column-accurately because the spacing/origin tuples contain
# unquoted commas; the logs carry the same measurements verbatim.
#
# Run:
#   venv/bin/python construction_time_table.py
#   venv/bin/python construction_time_table.py --results-dir /path --precision 1
# ============================================================
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("/home/adri/projects/TFM/optix_test_results3")

CONSTRUCTION_RE = re.compile(r"Construction time:\s*([0-9.]+)\s*ms")

DIST_MAP_RE        = re.compile(r"Adaptive distance map:\s*\S+\s*\(([0-9.]+) MB\)")
BRICK_SCENE_RE     = re.compile(r"Bricked-regions scene:\s*(\d+) primitives")
BRICK_RELEVANT_RE  = re.compile(r"Bricked-regions strategy:\s*(\d+)/(\d+) bricks are relevant")
OCTREE_RE          = re.compile(r"Octree:\s*(\d+) nodes,\s*([0-9.]+) MB,\s*(\d+) leaves,\s*(\d+) relevant")
OCTREE_REGIONS_RE  = re.compile(r"Octree-regions scene:\s*(\d+) leaf primitives")
NANOVDB_RE         = re.compile(r"NanoVDB grid:\s*([0-9.]+) MB on GPU,\s*(\d+) active voxels")

ORDER = ["manual", "optix", "adaptive", "bricked-regions", "octree",
         "octree-regions", "nanovdb"]

LATEX_INFO = {
    "manual":          ("---",            "Volume upload only"),
    "optix":           ("---",            "Volume upload + GAS"),
    "adaptive":        (r"$\sim$volume",  "Distance-map computation"),
    "bricked-regions": (r"$\sim$brick grid", "Classification + GAS"),
    "octree":          (r"$\sim$tree",    "Tree build (relevance flags)"),
    "octree-regions":  (r"$\sim$tree + GAS", "Tree build + leaf flattening"),
    "nanovdb":         ("grid-dependent", "VDB conversion + flattening"),
}

AUX_TEXTMAP = {
    "adaptive":        r"$\sim$volume (ch pkt. dist. map)",
    "octree":          r"$\sim$tree ($\sim$3.8 MB)",
}


def parse_log(path: Path) -> dict | None:
    """Return one record dict, or None if no construction time."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    m = CONSTRUCTION_RE.search(text)
    if not m:
        return None
    stem = path.stem
    if "--" in stem:
        strategy, variant = stem.split("--", 1)
    else:
        strategy, variant = stem, ""
    record = {
        "strategy": strategy,
        "variant": variant,
        "build_time_ms": float(m.group(1)),
        "series": path.parent.name,
        "file": path.name,
        "aux": {},
    }
    if "Adaptive distance map:" in text:
        mm = DIST_MAP_RE.search(text)
        if mm:
            record["aux"]["dist_map_mb"] = float(mm.group(1))
    if "Bricked-regions scene:" in text:
        mm = BRICK_SCENE_RE.search(text)
        if mm:
            record["aux"]["brick_primitives"] = int(mm.group(1))
    if "Bricked-regions strategy:" in text:
        mm = BRICK_RELEVANT_RE.search(text)
        if mm:
            record["aux"]["bricks"] = f"{mm.group(1)}/{mm.group(2)}"
    if "Octree-regions scene:" in text:
        mm = OCTREE_REGIONS_RE.search(text)
        if mm:
            record["aux"]["region_primitives"] = int(mm.group(1))
    if "Octree:" in text:
        mm = OCTREE_RE.search(text)
        if mm:
            record["aux"]["octree"] = (f"{mm.group(1)} nodes, {mm.group(2)} MB, "
                                       f"{mm.group(3)} leaves, {mm.group(4)} relevant")
    if "NanoVDB grid:" in text:
        mm = NANOVDB_RE.search(text)
        if mm:
            record["aux"]["nanovdb"] = f"{mm.group(1)} MB, {mm.group(2)} voxels"
    return record


def median_str(s: pd.Series, precision: int) -> str:
    return f"{np.median(s):,.{precision}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build construction-time table for the thesis")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--precision", type=int, default=1,
                    help="decimal places in the LaTeX numbers (default 1)")
    ap.add_argument("--mode", choices=["core", "all"], default="core",
                    help="'core' (default) excludes *-amon variants, whose build cost "
                         "is dominated by the adaptive distance-map step that the "
                         "thesis does not attribute to those strategies; "
                         "'all' uses every variant/dataset")
    args = ap.parse_args()

    logs = sorted(args.results_dir.rglob("*.log"))
    records = []
    for log in logs:
        rec = parse_log(log)
        if rec is None:
            print(f"  [skip] {log.parent.name}/{log.name} (no Construction time)")
            continue
        records.append(rec)

    if not records:
        raise SystemExit(f"No logs with Construction time found under {args.results_dir}")

    df = pd.DataFrame(records)

    print("=" * 78)
    print(f"Construction-time measurements parsed from {args.results_dir}")
    print(f"  {len(df)} valid logs out of {len(logs)} *.log files "
          f"({len(df.apply(lambda r: r['series'], axis=1).unique())} datasets)")
    print("=" * 78)
    print()

    # ---- per-strategy aggregate (all variants, all datasets) ----
    agg = df.groupby("strategy")["build_time_ms"].agg(["count", "median", "min", "max"])
    agg = agg.reindex(ORDER)  # thesis row order
    print("Per-strategy aggregate: median across ALL variants/datasets of the strategy")
    print(f"{'strategy':16s} {'n':>3s} {'median(ms)':>12s} {'min':>10s} {'max':>10s}")
    for strat in ORDER:
        r = agg.loc[strat]
        print(f"{strat:16s} {int(r['count']):3d} "
              f"{r['median']:10,.1f} {r['min']:10,.1f} {r['max']:10,.1f}")
    print()

    # ---- core-variant aggregate: pure behavior of each strategy ----
    #  amon variants embed the adaptive distance-map build, so their cost
    #  is dominated by a step that the limiting-factor column does not name
    #  for bricked-regions/octree. Report the amoff-only median as well.
    core = df[~df["variant"].str.contains("amon", na=False)].copy()
    if core.empty:
        core = df
    agg_core = core.groupby("strategy")["build_time_ms"].agg(["count", "median"])
    agg_core = agg_core.reindex(ORDER)
    print("Core-variant aggregate (excludes *-amon distance-map builds; "
          "same as overall for strategies without amode):")
    for strat in ORDER:
        r = agg_core.loc[strat]
        print(f"{strat:16s} {int(r['count']):3d} {r['median']:10,.1f} ms")
    print()

    # ---- per-variant median matrix ----
    print("Per-variant median build time (ms):")
    variants = ["eps001", "eps005", "eps010", "eps020"]
    for strat in ORDER:
        sub = df[df["strategy"] == strat]
        if sub.empty:
            continue
        print(f"  {strat}")
        for variant, g in sub.groupby("variant"):
            print(f"    {variant:28s} {np.median(g['build_time_ms']):10,.1f}  "
                  f"(n={len(g)})")
    print()

    # ---- auxiliary-structure sizes observed in the logs ----
    print("Auxiliary-structure sizes seen in the logs (informational, "
          "for the Aux. size column):")
    for strat in ORDER:
        sub = df[df["strategy"] == strat]
        vals = set()
        for rec in records:
            if rec["strategy"] == strat:
                for key, v in rec["aux"].items():
                    vals.add(f"{key}={v}")
        if vals:
            print(f"  {strat:16s} {', '.join(sorted(vals))}")

    # ---- LaTeX snippet ----
    def fmt(v: float) -> str:
        return f"\\texttt{{{v:,.{args.precision}f}}}"

    def emit(label: str, table: pd.DataFrame) -> None:
        print()
        print("=" * 78)
        print(f"LaTeX rows (median build_time_ms) -- {label}")
        print("=" * 78)
        print("% Replace the \\texttt{TODO} cells in tab:construction")
        print(r"\begin{tabular}{l r r l}")
        print(r"    \textbf{Strategy} & \textbf{build\_time\_ms} "
              r"& \textbf{Aux. size} & \textbf{Limiting factors} \\")
        print(r"    \midrule")
        for strat in ORDER:
            r = table.loc[strat]
            aux, factor = LATEX_INFO[strat]
            print(f"    \\texttt{{{strat}}}"
                  f" & {fmt(r['median'])}"
                  f" & {aux}"
                  f" & {factor} \\\\")
        print(r"    \bottomrule")
        print(r"\end{tabular}")

    # core-variant table merged back into full index
    agg_core_full = agg.copy()
    agg_core_full["median"] = agg_core["median"]

    emit("median excluding *-amon builds", agg_core_full)

    if args.mode == "all":
        emit("overall median (all variants & datasets)", agg)


if __name__ == "__main__":
    main()