# ============================================================
# D12 — Strategy validity classification (3-tier)
#   Classifies every run from ../optix_test_results2/results.csv
#   into VALID / ACCEPTABLE / INVALID based on foreground-masked
#   metrics vs the dataset's manual reference.
#
#   Thresholds (lenient, PSNR+SSIM primary):
#       valid:       PSNR >= 25 dB  AND  SSIM >= 0.90
#       acceptable:  PSNR >= 20 dB  AND  SSIM >= 0.80  (and not valid)
#       invalid:     everything else
#
#   Strict adds: MAE <= 3.0 / 8.0 and %changed <= 25 / 50.
#
#   Outputs (plots/):
#     d12_validity_classification.csv   per-run with tier
#     d12_validity_summary.csv          per strategy x epsilon
#     d12_validity_heatmap.png          strategy x epsilon, 3-tier %
#     d12_validity_by_epsilon.png       stacked bar per epsilon
#     d12_validity_top_strategies.png   ranked by max valid+acceptable
#
#   Usage: venv/bin/python d12_strategy_validity.py
# ============================================================
import re
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_ROOT = Path("/home/adri/projects/TFM/optix_test_results2")
RESULTS_CSV = RESULTS_ROOT / "results.csv"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)
CACHE_CSV = PLOT_DIR / "d11_fidelity_cache.csv"

EPSILONS = [0.01, 0.05, 0.1, 0.2]

# ── thresholds ───────────────────────────────────────────────
PSNR_VALID = 25.0
PSNR_ACCEPT = 20.0
SSIM_VALID = 0.90
SSIM_ACCEPT = 0.80
MAE_VALID = 3.0
MAE_ACCEPT = 8.0
PCT_VALID = 25.0
PCT_ACCEPT = 50.0

MODE_ORDER = ["manual", "optix", "adaptive",
              "bricked-regions", "octree", "octree-regions", "nanovdb"]


# ── helpers ──────────────────────────────────────────────────
def parse_strategy(s):
    if s in ("manual", "optix"):
        return s, None, {}
    m = re.match(r"^(.*)_(eps\d{3})((?:-[^-]+)*)$", s)
    if not m:
        return s, None, {}
    mode, eps_code = m.group(1), m.group(2)
    params = {}
    for part in m.group(3).lstrip("-").split("-"):
        if part.startswith("brick"):
            params["brick"] = part[5:]
        elif part.startswith("leaf"):
            params["leaf"] = part[4:]
        elif part.startswith("am"):
            params["am"] = part[2:]
        elif part in ("nearest", "trilinear"):
            params["sampler"] = part
    return mode, eps_code, params


def group_label(mode, params):
    if not params:
        return mode
    parts = []
    if "brick" in params:
        parts.append(f"brick{params['brick']}")
    if "leaf" in params:
        parts.append(f"leaf{params['leaf']}")
    if "sampler" in params:
        parts.append(params["sampler"])
    if "am" in params:
        parts.append(f"am-{params['am']}")
    return f"{mode} | {' '.join(parts)}"


def group_key(mode, params):
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else 99, mode,
            frozenset(params.items()))


def classify_lenient(psnr, ssim):
    """Return tier string from primary metrics (lenient rule)."""
    if psnr >= PSNR_VALID and ssim >= SSIM_VALID:
        return "valid"
    if psnr >= PSNR_ACCEPT and ssim >= SSIM_ACCEPT:
        return "acceptable"
    return "invalid"


def classify_strict(psnr, ssim, mae, pct):
    """Return tier string including secondary metrics."""
    if psnr >= PSNR_VALID and ssim >= SSIM_VALID and mae <= MAE_VALID and pct <= PCT_VALID:
        return "valid"
    if psnr >= PSNR_ACCEPT and ssim >= SSIM_ACCEPT and mae <= MAE_ACCEPT and pct <= PCT_ACCEPT:
        return "acceptable"
    return "invalid"


# ── main ─────────────────────────────────────────────────────
def main():
    if not CACHE_CSV.exists():
        sys.exit(f"Cache missing — run d11 first: {CACHE_CSV}")

    raw = open(RESULTS_CSV).read()
    fixed = re.sub(r"(\([^)]*,[^)]*\))", lambda m: f'"{m.group(1)}"', raw)
    res = pd.read_csv(io.StringIO(fixed))
    cache = pd.read_csv(CACHE_CSV)

    merged = cache.merge(
        res[["snapshot_path", "epsilon", "strategy", "mode"]],
        left_on="snapshot", right_on="snapshot_path",
    )
    print(f"{len(merged)} runs")

    # ── classify each run ────────────────────────────────────
    recs = []
    for _, r in merged.iterrows():
        mode, _eps_code, params = parse_strategy(r["strategy"])
        tier_l = classify_lenient(r["psnr"], r["ssim"])
        tier_s = classify_strict(r["psnr"], r["ssim"], r["mae"], r["pct_changed"])
        recs.append({
            "dataset": r["snapshot_path"].rsplit("/", 2)[-2],
            "strategy": r["strategy"],
            "label": group_label(mode, params),
            "group_order": group_key(mode, params),
            "mode": mode,
            "epsilon": r["epsilon"],
            "psnr": r["psnr"], "ssim": r["ssim"],
            "mae": r["mae"], "pct_changed": r["pct_changed"],
            "tier": tier_l, "tier_strict": tier_s,
        })
    df = pd.DataFrame(recs)

    df.to_csv(PLOT_DIR / "d12_validity_classification.csv", index=False)
    print("  saved d12_validity_classification.csv")

    # ── summary per (strategy, epsilon) ──────────────────────
    def count_tiers(sub):
        n = len(sub)
        nv = (sub["tier"] == "valid").sum()
        na = (sub["tier"] == "acceptable").sum()
        ni = (sub["tier"] == "invalid").sum()
        nvs = (sub["tier_strict"] == "valid").sum()
        nas = (sub["tier_strict"] == "acceptable").sum()
        nis = (sub["tier_strict"] == "invalid").sum()
        return pd.Series({
            "n": n,
            "n_valid": nv, "pct_valid": 100 * nv / n,
            "n_acceptable": na, "pct_acceptable": 100 * na / n,
            "n_invalid": ni, "pct_invalid": 100 * ni / n,
            "n_valid_or_acceptable": nv + na,
            "pct_valid_or_acceptable": 100 * (nv + na) / n,
            "n_valid_s": nvs, "pct_valid_s": 100 * nvs / n,
            "n_acceptable_s": nas, "pct_acceptable_s": 100 * nas / n,
            "n_invalid_s": nis, "pct_invalid_s": 100 * nis / n,
        })

    summary = (df.groupby(["label", "group_order", "epsilon"])
               .apply(count_tiers, include_groups=False)
               .reset_index()
               .sort_values(["group_order", "epsilon"]))
    summary.to_csv(PLOT_DIR / "d12_validity_summary.csv", index=False)
    print("  saved d12_validity_summary.csv")

    # ── console breakdown ────────────────────────────────────
    tot = len(df)
    nv = (df["tier"] == "valid").sum()
    na = (df["tier"] == "acceptable").sum()
    ni = (df["tier"] == "invalid").sum()
    print(f"\n=== overall: {nv} valid + {na} acceptable + {ni} invalid "
          f"out of {tot} ===")
    print(f"    {100*nv/tot:.1f}% valid  {100*na/tot:.1f}% acceptable  "
          f"{100*ni/tot:.1f}% invalid")
    print("\n=== per-epsilon tier breakdown ===")
    for e, gg in df.groupby("epsilon"):
        ev = (gg["tier"] == "valid").sum()
        ea = (gg["tier"] == "acceptable").sum()
        ei = (gg["tier"] == "invalid").sum()
        tot_e = len(gg)
        print(f"  eps {e:>4}: {ev:4d} valid  {ea:4d} acceptable  {ei:4d} invalid  "
              f"({100*ev/tot_e:5.1f}% + {100*ea/tot_e:5.1f}% = {100*(ev+ea)/tot_e:5.1f}% usable)")

    # ═══════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════
    labels = (df.drop_duplicates("label")
              .sort_values("group_order")["label"].tolist())
    eps_cols = EPSILONS
    tier_colors = {"valid": "#2e8b57", "acceptable": "#e6b800", "invalid": "#b22222"}

    # ── figure 1: heatmap (lenient + strict panels) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.52 * len(labels) + 3.5),
                             gridspec_kw={"wspace": 0.25})

    for ax, tier_col, title in zip(
        axes,
        ["pct_valid_or_acceptable", "pct_valid"],
        ["% valid + acceptable", "% valid only"],
    ):
        piv = summary.pivot(index="label", columns="epsilon", values=tier_col)
        piv = piv.reindex(labels).reindex(columns=eps_cols)
        vmin, vmax = (0, 100)
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(eps_cols)))
        ax.set_xticklabels([f"eps={e}" for e in eps_cols], fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}%",
                            ha="center", va="center", fontsize=6,
                            color="black" if 20 < v < 80 else "white")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03, label="% datasets")

    fig.suptitle("Strategy validity per epsilon (of 47 datasets) — "
                 f"valid: PSNR≥{PSNR_VALID} & SSIM≥{SSIM_VALID}  |  "
                 f"acceptable: PSNR≥{PSNR_ACCEPT} & SSIM≥{SSIM_ACCEPT}",
                 fontsize=11)
    fig.subplots_adjust(left=0.17, right=0.88, top=0.91, bottom=0.06,
                        wspace=0.25)
    out = PLOT_DIR / "d12_validity_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

    # ── figure 2: 3-tier stacked bar per epsilon ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, tier_key, title in zip(
        axes,
        ["tier", "tier_strict"],
        ["lenient (PSNR+SSIM)", "strict (+ MAE + %changed)"],
    ):
        bottoms = np.zeros(len(eps_cols))
        for tier, color in [("valid", "#2e8b57"), ("acceptable", "#e6b800"),
                            ("invalid", "#b22222")]:
            vals = np.array([
                100 * ((df[df["epsilon"] == e][tier_key] == tier).sum()
                       / len(df[df["epsilon"] == e]))
                for e in eps_cols
            ])
            ax.bar(eps_cols, vals, bottom=bottoms, color=color,
                   label=tier, edgecolor="white", linewidth=0.5)
            # annotate if > 3%
            for j, v in enumerate(vals):
                if v > 3:
                    ax.text(j, bottoms[j] + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white" if tier == "valid" else "black")
            bottoms += vals
        ax.set_ylabel("% of runs")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Overall tier breakdown per epsilon — all 4,042 runs", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.12, wspace=0.15)
    out = PLOT_DIR / "d12_validity_by_epsilon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

    # ── figure 3: ranked strategies (valid+acceptable) ───────
    best = (summary.sort_values("pct_valid_or_acceptable", ascending=False)
            .drop_duplicates("label"))
    best["eps_str"] = best["epsilon"].map(lambda e: f"eps={e}")
    best = best.sort_values("pct_valid_or_acceptable")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(best) + 3))
    y = np.arange(len(best))
    eps_cmap = {0.01: "#1f77b4", 0.05: "#ff7f0e", 0.1: "#2ca02c", 0.2: "#d62728"}
    bar_colors = [eps_cmap.get(e, "#999") for e in best["epsilon"]]
    ax.barh(y, best["pct_valid_or_acceptable"].values, color=bar_colors,
            edgecolor="white", linewidth=0.4, height=0.7)
    # show valid portion in darker shade
    for yi, row in best.iterrows():
        v = row["pct_valid"]
        va = row["pct_valid_or_acceptable"]
        if v > 1:
            ax.barh(y[yi == best.index][0], v, color="#1a5e1a",
                    edgecolor="none", height=0.7)
        label = f"{va:.0f}% (valid {v:.0f}%)  @ {row['eps_str']}"
        ax.text(0.5, list(best.index).index(yi), label,
                va="center", fontsize=6, color="white" if va > 15 else "black")
    ax.set_yticks(y)
    ax.set_yticklabels(best["label"].values, fontsize=6.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% datasets (valid + acceptable) at best epsilon")
    ax.set_title("Strategies ranked by maximum valid+acceptable %\n"
                 f"dark = valid  light = acceptable  |  color = best epsilon")
    handles = [plt.Rectangle((0, 0), 1, 1, color=eps_cmap[e])
               for e in EPSILONS]
    ax.legend(handles, [f"eps={e}" for e in EPSILONS], fontsize=7,
              title="best epsilon", loc="lower right")
    fig.subplots_adjust(left=0.20, right=0.92, top=0.93, bottom=0.07)
    out = PLOT_DIR / "d12_validity_top_strategies.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()