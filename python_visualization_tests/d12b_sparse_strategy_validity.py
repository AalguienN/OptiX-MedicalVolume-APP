# ============================================================
# D12b — Validity per sparse strategy (family-level)
#   Computes % of runs classified as valid / acceptable /
#   usable (valid+acceptable) / invalid for each of the 5
#   sparse strategies (adaptive, bricked-regions, octree,
#   octree-regions, nanovdb), overall and per epsilon.
#
#   Reads d12_validity_classification.csv (produced by
#   d12_strategy_validity.py).
#
#   Outputs (plots/):
#     d12b_sparse_strategy_summary.csv   family x epsilon tiers
#     d12b_sparse_strategy_overall.csv   family, all epsilons pooled
#     d12b_sparse_validity_stack.png     family x epsilon, stacked bars
#
#   Usage: venv/bin/python d12b_sparse_strategy_validity.py
# ============================================================
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent / "plots"
CLASS_CSV = PLOT_DIR / "d12_validity_classification.csv"

EPSILONS = [0.01, 0.05, 0.1, 0.2]

# the 5 sparse strategies, canonical order
SPARSE_MODES = ["adaptive", "bricked-regions", "octree",
                "octree-regions", "nanovdb"]

TIER_COLORS = {"valid": "#2e8b57", "acceptable": "#e6b800", "invalid": "#b22222"}


def summarize(sub):
    n = len(sub)
    nv = int((sub["tier"] == "valid").sum())
    na = int((sub["tier"] == "acceptable").sum())
    ni = int((sub["tier"] == "invalid").sum())
    return {
        "n": n,
        "n_valid": nv, "pct_valid": 100 * nv / n,
        "n_acceptable": na, "pct_acceptable": 100 * na / n,
        "n_usable": nv + na, "pct_usable": 100 * (nv + na) / n,
        "n_invalid": ni, "pct_invalid": 100 * ni / n,
    }


def main():
    if not CLASS_CSV.exists():
        sys.exit(f"Missing {CLASS_CSV} — run d12_strategy_validity.py first.")
    df = pd.read_csv(CLASS_CSV)
    for m in SPARSE_MODES:
        if m not in set(df["mode"]):
            sys.exit(f"Mode '{m}' not in classification CSV.")

    sparse = df[df["mode"].isin(SPARSE_MODES)]

    # ── summary per (family, epsilon) ────────────────────────
    rows = []
    for mode in SPARSE_MODES:
        sub_mode = sparse[sparse["mode"] == mode]
        for e in EPSILONS:
            sub = sub_mode[sub_mode["epsilon"] == e]
            row = {"mode": mode, "epsilon": e}
            row.update(summarize(sub))
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(PLOT_DIR / "d12b_sparse_strategy_summary.csv", index=False)
    print("  saved d12b_sparse_strategy_summary.csv")

    # ── overall (all epsilons pooled) per family ─────────────
    overall = []
    for mode in SPARSE_MODES:
        sub = sparse[sparse["mode"] == mode]
        row = {"mode": mode}
        row.update(summarize(sub))
        overall.append(row)
    overall_df = pd.DataFrame(overall)
    overall_df.to_csv(PLOT_DIR / "d12b_sparse_strategy_overall.csv", index=False)
    print("  saved d12b_sparse_strategy_overall.csv")

    # ── console ──────────────────────────────────────────────
    print("\n=== % per sparse strategy (all epsilons pooled) ===")
    print(f"{'strategy':<16}{'n':>5}{'valid':>9}{'accept':>9}{'usable':>9}{'invalid':>9}")
    for _, r in overall_df.iterrows():
        print(f"{r['mode']:<16}{r['n']:>5d}"
              f"{r['pct_valid']:>8.1f}%{r['pct_acceptable']:>8.1f}%"
              f"{r['pct_usable']:>8.1f}%{r['pct_invalid']:>8.1f}%")

    print("\n=== per-epsilon, pooled across the 5 sparse strategies ===")
    for e, sub in sparse.groupby("epsilon"):
        s = summarize(sub)
        print(f"  eps {e:>4}: valid {s['pct_valid']:5.1f}%  "
              f"accept {s['pct_acceptable']:5.1f}%  "
              f"usable {s['pct_usable']:5.1f}%  "
              f"invalid {s['pct_invalid']:5.1f}%")

    # ══════════════════════════════════════════════════════════
    # figure: heatmap of 4 tier metrics (rows) x epsilon (cols),
    #         one panel per strategy
    # ══════════════════════════════════════════════════════════
    metrics = [
        ("pct_valid", "valid", TIER_COLORS["valid"]),
        ("pct_acceptable", "acceptable", TIER_COLORS["acceptable"]),
        ("pct_usable", "usable (V+A)", "#1a7e4a"),
        ("pct_invalid", "invalid", TIER_COLORS["invalid"]),
    ]
    n_mode = len(SPARSE_MODES)
    fig, axes = plt.subplots(1, n_mode, figsize=(3.1 * n_mode, 2.6),
                             squeeze=False)
    for mi, mode in enumerate(SPARSE_MODES):
        ax = axes[0, mi]
        sub = summary[summary["mode"] == mode].set_index("epsilon")
        # stacked vertical bars per epsilon
        eps_pos = np.arange(len(EPSILONS))
        bottoms = np.zeros(len(EPSILONS))
        for col, label, color in metrics:
            vals = sub.reindex(EPSILONS)[col].values
            ax.bar(eps_pos, vals, 0.6, bottom=bottoms, color=color,
                   label=label, edgecolor="white", linewidth=0.6)
            bottoms += vals
        ax.set_xticks(eps_pos)
        ax.set_xticklabels([f"eps={e}" for e in EPSILONS], fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_title(mode.replace("-", "-\n"), fontsize=8)
        ax.tick_params(labelsize=7, labelleft=False)
        if mi == 0:
            ax.set_ylabel("% of runs", fontsize=7.5)
            ax.tick_params(labelleft=True)
        ax.legend(fontsize=5.5, loc="upper right", ncol=1,
                  frameon=True, framealpha=0.8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    fig.suptitle("Validity tier split per sparse strategy x epsilon",
                 fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.10,
                        wspace=0.18)
    out = PLOT_DIR / "d12b_sparse_validity_stack.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()