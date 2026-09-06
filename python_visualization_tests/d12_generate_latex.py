"""Generate a LaTeX table from d12_validity_summary.csv."""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "plots/d12_validity_summary.csv"
OUT = Path(__file__).resolve().parent / "plots/d12_validity_table.tex"

s = pd.read_csv(CSV)
EPS = [0.01, 0.05, 0.1, 0.2]
# sort by group order then epsilon
s = s.sort_values(["group_order", "epsilon"])

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Validity classification per strategy $\times$ $\epsilon$ "
             r"(foreground-masked vs.\ manual, 47 datasets). "
             r"V = valid (PSNR$\geq$25 dB, SSIM$\geq$0.90), "
             r"A = acceptable (PSNR$\geq$20 dB, SSIM$\geq$0.80), "
             r"I = invalid. Bold = usable (V+A) $\geq$ 90\%.}")
lines.append(r"\label{tab:validity}")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{3pt}")
lines.append(r"\begin{tabular}{l cccc}")
lines.append(r"\toprule")
lines.append(r"\textbf{Strategy} & \multicolumn{4}{c}{\textbf{\% of 47 datasets (V / A / I)}} \\")
lines.append(r"\cmidrule(lr){2-5}")
lines.append(r" & $\epsilon{=}0.01$ & $\epsilon{=}0.05$ & $\epsilon{=}0.10$ & $\epsilon{=}0.20$ \\")
lines.append(r"\midrule")

# group by strategy label (keep one row per strategy with 4 epsilon columns)
strategies = s.drop_duplicates("label").sort_values("group_order")["label"]
for strat in strategies:
    sub = s[s["label"] == strat]
    row = f"{strat.replace('|', '/')} "
    for e in EPS:
        erow = sub[sub["epsilon"] == e]
        if erow.empty:
            row += " & ---"
            continue
        v = erow["pct_valid"].values[0]
        a = erow["pct_acceptable"].values[0]
        ua = v + a
        iv = max(0.0, 100.0 - v - a)
        # color: green if >=90% usable, yellow if >=50%, red otherwise
        if ua >= 90:
            cell = f"\\textbf{{{v:.0f}}} / {a:.0f} / {iv:.0f}"
        elif ua >= 50:
            cell = f"{v:.0f} / {a:.0f} / {iv:.0f}"
        else:
            cell = f"{v:.0f} / {a:.0f} / \\textcolor{{red}}{{{iv:.0f}}}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

# add overall row
lines.append(r"\midrule")
overall_row = r"\textbf{Overall} "
for e in EPS:
    es = s[s["epsilon"] == e]
    tot = es["n"].sum()
    v = 100 * es["n_valid"].sum() / tot
    a = 100 * es["n_acceptable"].sum() / tot
    overall_row += f" & \\textbf{{{v:.0f}}} / {a:.0f} / {100-v-a:.0f}"
overall_row += r" \\"
lines.append(overall_row)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")
lines.append("")
lines.append(r"% Usage in LaTeX:")
lines.append(r"% \input{d12_validity_table.tex}")
lines.append(r"% Requires: \usepackage{booktabs}  \usepackage{xcolor}")

OUT.write_text("\n".join(lines))
print(f"  saved {OUT}")
