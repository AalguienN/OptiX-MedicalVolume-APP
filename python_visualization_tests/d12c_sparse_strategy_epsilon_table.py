# ============================================================
# D12c — Cross-product LaTeX table: strategy x epsilon validity
#   20 rows = 5 sparse strategies x 4 epsilons, each cell as
#   "V / A / I" (% valid / acceptable / invalid of the family's
#   runs for that epsilon).
#
#   Reads d12b_sparse_strategy_summary.csv (produced by
#   d12b_sparse_strategy_validity.py).
#
#   Output (plots/): d12c_sparse_strategy_epsilon_table.tex
#
#   Usage: venv/bin/python d12c_sparse_strategy_epsilon_table.py
# ============================================================
import io
import re
from pathlib import Path

import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent / "plots"
SUMMARY_CSV = PLOT_DIR / "d12b_sparse_strategy_summary.csv"
OUT = PLOT_DIR / "d12c_sparse_strategy_epsilon_table.tex"

EPSILONS = [0.01, 0.05, 0.1, 0.2]
SPARSE_MODES = ["adaptive", "bricked-regions", "octree",
                "octree-regions", "nanovdb"]


def fmt_cell(v, a, iv, usable_pct):
    """Format V / A / I with bold/red emphasis."""
    if usable_pct >= 90:
        return f"\\textbf{{{v:.0f}}} / {a:.0f} / {iv:.0f}"
    if usable_pct >= 50:
        return f"{v:.0f} / {a:.0f} / {iv:.0f}"
    return f"{v:.0f} / {a:.0f} / \\textcolor{{red}}{{{iv:.0f}}}"


def main():
    s = pd.read_csv(SUMMARY_CSV)
    s = s[s["mode"].isin(SPARSE_MODES)]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Validity classification cross-product of the 5 sparse "
        r"strategies $\times$ $\epsilon$ (foreground-masked vs.\ manual). "
        r"V = valid (PSNR$\geq$25\,dB, SSIM$\geq$0.90), "
        r"A = acceptable (PSNR$\geq$20\,dB, SSIM$\geq$0.80), "
        r"I = invalid. Bold = usable (V+A) $\geq$ 90\%; red = unusable "
        r"(V+A $<$ 50\%).}")
    lines.append(r"\label{tab:validity_cross}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Strategy} & \textbf{$\epsilon$} & "
                 r"\textbf{\% of runs (V / A / I)} \\")
    lines.append(r"\midrule")

    for mode in SPARSE_MODES:
        sub = s[s["mode"] == mode].set_index("epsilon")
        for e in EPSILONS:
            if e not in sub.index:
                continue
            r = sub.loc[e]
            v, a = r["pct_valid"], r["pct_acceptable"]
            iv = r["pct_invalid"]
            cell = fmt_cell(v, a, iv, r["pct_usable"])
            strat_name = mode if e == EPSILONS[0] else ""
            eps_label = f"{e:g}"  # strip trailing zeros
            eps_tex = r"$\epsilon={}$".format(eps_label)
            lines.append(f"{strat_name} & {eps_tex} & {cell} \\\\")
        if mode != SPARSE_MODES[-1]:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")
    lines.append(r"% Usage: \input{d12c_sparse_strategy_epsilon_table.tex}")
    lines.append(r"% Requires: \usepackage{booktabs} \usepackage{xcolor}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"  saved {OUT}")

    print("\n=== preview ===")
    print(f"{'strategy':<18}{'eps':>8}{'valid':>9}{'accept':>9}{'invalid':>9}")
    for mode in SPARSE_MODES:
        sub = s[s["mode"] == mode].set_index("epsilon")
        for e in EPSILONS:
            r = sub.loc[e]
            print(f"{mode:<18}{e:>8.2f}"
                  f"{r['pct_valid']:>8.1f}%{r['pct_acceptable']:>8.1f}%"
                  f"{r['pct_invalid']:>8.1f}%")


if __name__ == "__main__":
    main()