# ============================================================
# Effect of the swept parameters (subsec "Effect of the swept
# parameters") from optix_test_results2/results.csv.
#
# For each dependent variable of the experimental matrix it
# reports the *median over the test set (47 datasets)* of the
# per-dataset speedup (manual_render / strategy_render), exactly
# as in the rendering-performance table:
#   * brick-size   -> bricked-regions
#   * leaf-size    -> octree, octree-regions
#   * adaptive-march -> bricked-regions, octree, octree-regions
#
# A "cell" is one dataset x hold-parameters: the render time is
# first median-aggregated across the swept-out variants (am, leaf,
# brick), then the per-dataset speedup and finally the median over
# the test set are taken (mirrors table tab:perf_per_eps).
#
# Run:
#   venv/bin/python param_sweep_effects.py
#   venv/bin/python param_sweep_effects.py --results-dir /path --save
# ============================================================
import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("/home/adri/projects/TFM/optix_test_results2")
EPSILONS = [0.01, 0.05, 0.10, 0.20]


def load(csv_path: Path) -> pd.DataFrame:
    raw = open(csv_path).read()
    fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
    return pd.read_csv(io.StringIO(fixed))


def parse_params(strategy: str) -> dict:
    """Extract {brick, leaf, am, sampler} from a results.csv strategy label."""
    if strategy in ("manual", "optix"):
        return {}
    m = re.match(r"^(.*)_(eps\d{3})((?:-[^-]+)*)$", strategy)
    if not m:
        return {}
    d = {}
    for part in m.group(3).lstrip("-").split("-"):
        if part.startswith("brick"):
            d["brick"] = int(part[5:])
        elif part.startswith("leaf"):
            d["leaf"] = int(part[4:])
        elif part.startswith("am"):
            d["am"] = part[2:]
        elif part in ("nearest", "trilinear"):
            d["sampler"] = part
    return d


def cell_summary(df: pd.DataFrame, hold: list, drop: list,
                 manual: pd.Series) -> pd.DataFrame:
    """Median over the test set of per-dataset speedup for (hold).

    Render is median-averaged across the 'drop' variants inside each
    dataset x hold cell; speedup is computed per dataset against the
    manual baseline; the reported value is the median over datasets.
    """
    g = df.groupby(["dataset"] + hold + drop).agg(
        render=("render_mean_ms", "median"),
        fps=("fps_mean", "median"),
        skip=("skip_ratio_bounds", "median"),
    ).reset_index()
    g["speedup"] = g["dataset"].map(manual) / g["render"]
    return g.groupby(hold)[["speedup", "render", "fps", "skip"]].median()


def am_gain(df: pd.DataFrame, hold: list, manual: pd.Series) -> pd.Series:
    """Median per-dataset relative gain (in %) of am=on over am=off."""
    g = df.groupby(["dataset"] + hold + ["am"]).agg(
        render=("render_mean_ms", "median"),
    ).reset_index()
    g["speedup"] = g["dataset"].map(manual) / g["render"]
    wide = g.pivot_table(index=["dataset"] + hold, columns="am",
                         values="speedup")
    gain = (wide["on"] - wide["off"]) / wide["off"] * 100.0
    return gain.groupby(hold).median().rename("gain")


def print_grid(res: pd.DataFrame, title: str, cols: list, row: str,
               value: str = "speedup") -> None:
    print(f"\n  {title}  [median of per-dataset {value}]")
    tab = res.pivot_table(index=row, columns="level", values=value)[cols]
    print(tab.round(3).to_string())
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(description="Effect of brick/leaf size and "
                                            "adaptive-march on rendering speedup")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--save", action="store_true",
                    help="also write the report to plots/param_sweep_effects.txt")
    args = ap.parse_args()

    df = load(args.results_dir / "results.csv")
    params = df["strategy"].map(parse_params)
    for col in ("brick", "leaf", "am", "sampler"):
        df[col] = [d.get(col) for d in params]
    manual = df[df["mode"] == "manual"].set_index("dataset")["render_mean_ms"]
    print(f"{len(df)} rows loaded across {df['dataset'].nunique()} datasets")

    out = []

    # ================== 1. BRICK SIZE (bricked-regions) ==================
    sub = df[df["mode"] == "bricked-regions"].copy()
    res = cell_summary(sub, hold=["epsilon", "brick"], drop=["am"], manual=manual)
    res = res.reset_index().rename(columns={"brick": "level"})
    res["level"] = res["level"].astype(int)
    print("\n" + "=" * 70)
    print("BRICK SIZE  (bricked-regions) -- median speedup over manual")
    print("=" * 70)
    tab = print_grid(res, "by epsilon", [8, 16, 32], "epsilon")

    overall = res.groupby("level")["speedup"].median()
    best = overall.idxmax()
    per_eps_best = res.loc[res.groupby("epsilon")["speedup"].idxmax()]
    print("  best brick per epsilon:", {f"{e:.2f}": int(b) for e, b in
          zip(per_eps_best["epsilon"], per_eps_best["level"])})
    print("  per-level median (collapsing eps):",
          {int(l): round(overall[l], 3) for l in overall.index})
    brick_best = per_eps_best
    brick_res = res.copy()
    out.extend([
        "BRICK SIZE (bricked-regions), median speedup per (eps, brick):",
        tab.round(3).to_string(),
        f"best brick overall: {best}",
    ])

    # ================== 2. LEAF SIZE (octree, octree-regions) =============
    print("\n" + "=" * 70)
    print("LEAF SIZE  (median speedup over manual)")
    print("=" * 70)
    live_best = {}
    leaf_res = {}
    for mode in ("octree", "octree-regions"):
        sub = df[df["mode"] == mode].copy()
        res = cell_summary(sub, hold=["epsilon", "leaf"], drop=["am"], manual=manual)
        res = res.reset_index().rename(columns={"leaf": "level"})
        res["level"] = res["level"].astype(int)
        tab = print_grid(res, f"{mode}", [4, 8, 16], "epsilon")
        per_eps_best = res.loc[res.groupby("epsilon")["speedup"].idxmax()]
        print("  best leaf per epsilon:",
              {f"{e:.2f}": int(b) for e, b in
               zip(per_eps_best["epsilon"], per_eps_best["level"])})
        overall = res.groupby("level")["speedup"].median().round(3)
        print("  per-level median (collapsing eps):",
              {int(l): overall[l] for l in overall.index})
        live_best[mode] = (per_eps_best, overall)
        leaf_res[mode] = res.copy()
        out.extend([f"LEAF SIZE {mode}:", tab.round(3).to_string(),
                    f"best leaf per eps: "
                    f"{dict(zip(per_eps_best['epsilon'].round(2), per_eps_best['level']))}"])
    sub = df[df["mode"].isin(["octree", "octree-regions"])].copy()
    res = cell_summary(sub, hold=["mode", "leaf"], drop=["am"], manual=manual)
    gap = res.xs("octree-regions", level="mode")["speedup"] / \
          res.xs("octree", level="mode")["speedup"]
    print("\n  octree-regions / octree speedup ratio per leaf (collapsing eps):")
    print("   " + gap.round(2).to_string())
    print(f"   -> octree-regions is {((gap.median()-1)*100):.0f}% faster "
          f"on average across leaf sizes")

    # ================== 3. ADAPTIVE MARCH =================================
    print("\n" + "=" * 70)
    print("ADAPTIVE MARCH  (gain of am=on over am=off, % per dataset)")
    print("=" * 70)
    am_notes = []
    am_res = {}
    for mode, param in (("bricked-regions", "brick"),
                        ("octree", "leaf"),
                        ("octree-regions", "leaf")):
        sub = df[df["mode"] == mode].copy()
        g = am_gain(sub, hold=["epsilon", param], manual=manual).reset_index()
        g = g.rename(columns={param: "level", "gain": "gain"})
        g["level"] = g["level"].astype(int)
        print(f"\n  {mode}  -- gain(%) of adaptive-march on, by eps x {param}:")
        tab = g.pivot_table(index="epsilon", columns="level", values="gain")
        print(tab.round(1).to_string())
        # overall gain and the small-vs-large parameter levels
        overall = g.groupby("level")["gain"].median()
        small, large = sorted(overall.index)[0], sorted(overall.index)[-1]
        print(f"    overall (collapse eps): "
              + ", ".join(f"{param}{l}={overall[l]:+.1f}%"
                          for l in sorted(overall.index)))
        print(f"    gain at {param}={small}: {overall[small]:+.1f}%  "
              f"at {param}={large}: {overall[large]:+.1f}%")
        am_notes.append((mode, param, overall))
        am_res[mode] = g.copy()
        out.extend([f"AM {mode}:", tab.round(1).to_string(),
                    f"overall gain {overall.round(1).to_string()}"])

    # ---- conclusion -------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSION  (measured best parameters per strategy)")
    print("=" * 70)

    def fmt_optima(per_eps_best):
        return ", ".join(f"{e:.2f}->{int(l)}" for e, l in
                         zip(per_eps_best["epsilon"], per_eps_best["level"]))

    print(f"  bricked-regions brick-size: optimum is epsilon-dependent: "
          f"{fmt_optima(brick_best)}")
    for mode, param, overall in am_notes:
        print(f"  {mode} adaptive-march ({param}): "
              + ", ".join(f"{param}{l}={overall[l]:+.1f}%"
                          for l in sorted(overall.index)))
    for mode in ("octree", "octree-regions"):
        per_eps_best, overall = live_best[mode]
        print(f"  {mode} leaf-size: optimum per eps {fmt_optima(per_eps_best)} "
              f"| per-level median "
              f"{dict((int(l), round(overall[l], 3)) for l in overall.index)}")

    latex = build_latex(brick_res, leaf_res, am_res)
    print("\n" + "=" * 78)
    print("LaTeX tables for the 'Effect of the swept parameters' subsection")
    print("=" * 78)
    print(latex)

    if args.save:
        plot_dir = Path(__file__).resolve().parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        fname = plot_dir / "param_sweep_effects.txt"
        fname.write_text("\n".join(out) + "\n")
        print(f"\n  saved {fname}")
        tex = plot_dir / "param_sweep_tables.tex"
        tex.write_text(latex)
        print(f"  saved {tex}")


def build_latex(brick_res, leaf_res, am_res) -> str:
    """Return the three LaTeX tables for the swept-parameters subsection."""
    eps = EPSILONS

    # ---- 1. brick size ----------------------------------------------
    t1 = brick_res.pivot_table(index="epsilon", columns="level",
                               values="speedup")[[8, 16, 32]]
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Median speedup over \texttt{manual} of "
                 r"\texttt{bricked-regions}, by brick size and $\epsilon$ "
                 r"(median over the 47 test datasets; best level in bold).}")
    lines.append(r"\label{tab:param_brick}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.3}")
    lines.append(r"\begin{tabular}{c r r r}")
    lines.append(r"\toprule")
    lines.append(r"$\epsilon$ & \textbf{8$^{3}$} & \textbf{16$^{3}$} & "
                 r"\textbf{32$^{3}$} \\")
    lines.append(r"\midrule")
    for e in eps:
        row = t1.loc[e]
        best = row.idxmax()
        cells = []
        for lv in (8, 16, 32):
            v = row[lv]
            cells.append((r"\textbf{%s}" % f"{v:.2f}") if lv == best
                         else f"{v:.2f}")
        lines.append(f"{e:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ---- 2. leaf size -----------------------------------------------
    oct_tab = leaf_res["octree"].pivot_table(index="epsilon", columns="level",
                                             values="speedup")[[4, 8, 16]]
    or_tab = leaf_res["octree-regions"].pivot_table(index="epsilon",
                                                    columns="level",
                                                    values="speedup")[[4, 8, 16]]
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Median speedup over \texttt{manual} by leaf size "
                 r"and $\epsilon$ for \texttt{octree} and \texttt{octree-regions}"
                 r" (best level per strategy in bold).}")
    lines.append(r"\label{tab:param_leaf}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.3}")
    lines.append(r"\begin{tabular}{c"
                 r" r r r"
                 r" r r r}")
    lines.append(r"\toprule")
    lines.append(r"$\epsilon$ & \multicolumn{3}{c}{\texttt{octree}} & "
                 r"\multicolumn{3}{c}{\texttt{octree-regions}} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r" & \textbf{4} & \textbf{8} & \textbf{16}"
                 r" & \textbf{4} & \textbf{8} & \textbf{16} \\")
    lines.append(r"\midrule")
    for e in eps:
        bo = oct_tab.loc[e].idxmax()
        bco = or_tab.loc[e].idxmax()
        cells = []
        for lv in (4, 8, 16):
            v = oct_tab.loc[e, lv]
            cells.append((r"\textbf{%s}" % f"{v:.2f}") if lv == bo else
                         f"{v:.2f}")
        for lv in (4, 8, 16):
            v = or_tab.loc[e, lv]
            cells.append((r"\textbf{%s}" % f"{v:.2f}") if lv == bco else
                         f"{v:.2f}")
        lines.append(f"{e:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    # ---- 3. adaptive march -------------------------------------------
    am_tabs = {m: am_res[m].pivot_table(index="epsilon", columns="level",
                                        values="gain")
               for m in ("bricked-regions", "octree", "octree-regions")}
    levels = {"bricked-regions": [8, 16, 32],
              "octree": [4, 8, 16], "octree-regions": [4, 8, 16]}
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Relative change of render time when "
                 r"\texttt{adaptive-march} is enabled (gain of \texttt{on} "
                 r"over \texttt{off}, median over the 47 test datasets; best "
                 r"level in bold). Negative values mean \texttt{adaptive-march}"
                 r" slows the strategy down.}")
    lines.append(r"\label{tab:param_am}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.3}")
    lines.append(r"\begin{tabular}{c"
                 r" r r r"
                 r" r r r"
                 r" r r r}")
    lines.append(r"\toprule")
    lines.append(r"$\epsilon$ & \multicolumn{3}{c}{\texttt{bricked-regions}} & "
                 r"\multicolumn{3}{c}{\texttt{octree}} & "
                 r"\multicolumn{3}{c}{\texttt{octree-regions}} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    lines.append(r" & \textbf{8} & \textbf{16} & \textbf{32}"
                 r" & \textbf{4} & \textbf{8} & \textbf{16}"
                 r" & \textbf{4} & \textbf{8} & \textbf{16} \\")
    lines.append(r"\midrule")
    for e in eps:
        cells = []
        for m in ("bricked-regions", "octree", "octree-regions"):
            row = am_tabs[m].loc[e]
            best = row.idxmax()
            for lv in levels[m]:
                v = row[lv]
                cells.append((r"\textbf{%+.1f\%%}" % v) if lv == best else
                             f"{v:+.1f}\\%")
        lines.append(f"{e:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()