#!/usr/bin/env bash
#
# run_tests3.sh — Run all rendering strategies with varied parameters
# across every TCIA dataset found in ~/Documents/TCIA, measuring the
# load-time construction time (build_time_ms) of each strategy.
#
# Results (CSV metrics + PPM snapshots) are stored in ../optix_test_results3/
# relative to the project root (i.e. beside this repo as a brother folder).
#
# Construction time requires the --build-time flag in optix_app: when passed,
# the binary times the whole load-time build (volume upload, transfer-function
# build, relevance classification, the per-strategy sparse representation and
# acceleration-structure construction, and pipeline setup, up to the render
# loop) and records it as an extra build_time_ms column in the aggregate
# results.csv row.
#
# Usage:
#   ./run_tests3.sh             # run all tests with construction-time measurement
#   ./run_tests3.sh --dry-run   # print commands without executing
#   ./run_tests3.sh --resume    # skip combinations that already have a CSV
#

set -euo pipefail

# ── resolve project root (where this script lives) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BINARY="$PROJECT_ROOT/build/bin/optix_app"

# ── output root (brother folder of the project) ─────────────────────────────
OUTPUT_ROOT="$(dirname "$PROJECT_ROOT")/optix_test_results3"

# ── TCIA data root ──────────────────────────────────────────────────────────
TCIA_ROOT="${TCIA_ROOT:-$HOME/Documents/TCIA}"

# ── CLI flags ───────────────────────────────────────────────────────────────
DRY_RUN=false
RESUME=false
FRAMES=200

usage() {
    cat <<EOF
Usage: $0 [options]

Runs every rendering strategy against every TCIA dataset (found automatically in
\$TCIA_ROOT, default ~/Documents/TCIA) with a cross-product sweep of parameter
dimensions. Each execution is one run; results are stored in the brother folder
../optix_test_results3/.

Version of run_tests2.sh that additionally measures the load-time construction
time of each strategy (via optix_app --build-time) and stores it as a
build_time_ms column in results.csv.

Strategies and their swept dimensions (all cross-products):
  manual                 (no params)
  optix                  (no params)
  adaptive-march         eps          ∈ {0.01, 0.05, 0.1, 0.2}
  bricked-regions        eps          ∈ {0.01, 0.05, 0.1, 0.2},
                         brick-size   ∈ {8, 16, 32},
                         adaptive-march ∈ {on, off}
  octree                 eps          ∈ {0.01, 0.05, 0.1, 0.2},
                         leaf-size    ∈ {4, 8, 16},
                         adaptive-march ∈ {on, off}
  octree-regions         eps          ∈ {0.01, 0.05, 0.1, 0.2},
                         leaf-size    ∈ {4, 8, 16},
                         adaptive-march ∈ {on, off}
  nanovdb                eps          ∈ {0.01, 0.05, 0.1, 0.2},
                         nanovdb-sampler ∈ {nearest, trilinear}

Notes:
  * Build time does not depend on the number of rendered frames. For a
    construction-only sweep (fast, no timing stats), pass --frames 1.
  * optix_app --build-time must be supported by the binary; rebuild with
    ./mkbuild.sh if results.csv lacks the build_time_ms column.

Options:
  -h, --help       Show this help and exit.
  --wipe           Delete the whole results folder (../optix_test_results3/) and
                   exit immediately. Wipes ONLY — no rendering; all other flags
                   are ignored. Run a fresh ./run_tests3.sh afterwards.
  --dry-run        Print the commands that would run, without executing them.
  --resume         Skip any configuration whose CSV already exists (useful to
                   continue after an interruption).
  --frames N       Frames per run (default 200). Lower = faster but noisier
                   timing stats; use higher for final measurements. Construction
                   time is independent of this value.

Environment:
  TCIA_ROOT        Directory scanned for manifest-*/series_* datasets
                   (default: \$HOME/Documents/TCIA).

Output (one row per execution):
  results.csv      Aggregate summary CSV at the output root — dataset, strategy,
                   mode, timings, fps, gpu mem, sparsity, traversal counters,
                   skip ratio, snapshot path, and (with --build-time) a
                   build_time_ms construction-time column.
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
        --wipe)
            rm -rf "$OUTPUT_ROOT"
            echo "Wiped: $OUTPUT_ROOT"
            exit 0
            ;;
        --dry-run)  DRY_RUN=true  ;;
        --resume)   RESUME=true   ;;
        --frames)   ;;  # handled below
        -*)         echo "Unknown flag: $arg"; exit 1 ;;
    esac
done
for i in $(seq 1 $#); do
    if [[ "${!i}" == "--frames" ]]; then
        j=$((i+1))
        FRAMES="${!j}"
    fi
done

# ── preflight checks ───────────────────────────────────────────────────────
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "       Build first with: ./mkbuild.sh"
    exit 1
fi

if [[ ! -d "$TCIA_ROOT" ]]; then
    echo "ERROR: TCIA root not found at $TCIA_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
LOGFILE="$OUTPUT_ROOT/test_run.log"

# ── discover datasets ───────────────────────────────────────────────────────
# Find every series directory: <manifest>/series_<N>
# Stored as relative paths from TCIA_ROOT for clean naming.
DATASETS=()
while IFS= read -r -d '' series_dir; do
    rel="${series_dir#"$TCIA_ROOT"/}"
    DATASETS+=("$rel")
done < <(find "$TCIA_ROOT" -mindepth 2 -maxdepth 2 -type d -name "series_*" -print0 | sort -z)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "ERROR: No series_* directories found under $TCIA_ROOT"
    exit 1
fi

echo "Found ${#DATASETS[@]} datasets under $TCIA_ROOT"
echo "Results will be saved to: $OUTPUT_ROOT"
echo "Frames per run: $FRAMES"
echo "Construction time measurement: on (--build-time)"
echo ""

# ── strategy configurations ─────────────────────────────────────────────────
# Each entry is a label followed by the CLI args for that configuration.
# Format: <label>|<args>|<file-tag>
# The label is used for naming; the file-tag is used for the CSV/PPM filename
# (must be filesystem-safe).

EPSILONS=(0.01 0.05 0.1 0.2)

# Label/tag helper: build an eps tag like "eps010" from a numeric epsilon.
eps_tag() {
    awk -v e="$1" 'BEGIN { printf "eps%03d", e*100 }'
}

STRATEGIES=()

# --- manual / optix: no extra params ---
STRATEGIES+=("manual|manual|manual")
STRATEGIES+=("optix|optix|optix")

# --- adaptive-march: sweep epsilon ---
for eps in "${EPSILONS[@]}"; do
    t=$(eps_tag "$eps")
    STRATEGIES+=("adaptive_$t|adaptive --epsilon $eps|adaptive--$t")
done

# --- bricked-regions: eps × brick-size × adaptive-march (cross-product) ---
for eps in "${EPSILONS[@]}"; do
    t=$(eps_tag "$eps")
    for bs in 8 16 32; do
        for am in on off; do
            STRATEGIES+=("bricked-regions_$t-brick$bs-am$am|bricked-regions --epsilon $eps --brick-size $bs --adaptive-march $am|bricked-regions--$t-brick$bs-am$am")
        done
    done
done

# --- octree: eps × leaf-size × adaptive-march (cross-product) ---
for eps in "${EPSILONS[@]}"; do
    t=$(eps_tag "$eps")
    for ls in 4 8 16; do
        for am in on off; do
            STRATEGIES+=("octree_$t-leaf$ls-am$am|octree --epsilon $eps --leaf-size $ls --adaptive-march $am|octree--$t-leaf$ls-am$am")
        done
    done
done

# --- octree-regions: eps × leaf-size × adaptive-march (cross-product) ---
for eps in "${EPSILONS[@]}"; do
    t=$(eps_tag "$eps")
    for ls in 4 8 16; do
        for am in on off; do
            STRATEGIES+=("octree-regions_$t-leaf$ls-am$am|octree-regions --epsilon $eps --leaf-size $ls --adaptive-march $am|octree-regions--$t-leaf$ls-am$am")
        done
    done
done

# --- nanovdb: eps × nanovdb-sampler (cross-product) ---
for eps in "${EPSILONS[@]}"; do
    t=$(eps_tag "$eps")
    for s in nearest trilinear; do
        STRATEGIES+=("nanovdb_$t-$s|nanovdb --epsilon $eps --nanovdb-sampler $s|nanovdb--$t-$s")
    done
done

echo "Strategy configurations: ${#STRATEGIES[@]}"
total=$((${#DATASETS[@]} * ${#STRATEGIES[@]}))
echo "Total test runs: $total"
echo ""

# ── run loop ────────────────────────────────────────────────────────────────
run_count=0
skip_count=0
fail_count=0
total_runs=$(( ${#DATASETS[@]} * ${#STRATEGIES[@]} ))

for dataset in "${DATASETS[@]}"; do
    # Derive a short, clean name for the dataset directory.
    # e.g. manifest-1788373825860_DICOM/series_3 → manifest-1788373825860_DICOM__series_3
    dataset_clean="${dataset//\//__}"
    dataset_dir="$OUTPUT_ROOT/$dataset_clean"
    mkdir -p "$dataset_dir"

    # --- dataset-level info (independent of strategy/epsilon) ---
    series_path="$TCIA_ROOT/$dataset"
    # Size on disk: total bytes of the .dcm files in the series directory.
    disk_bytes=$(find "$series_path" -name '*.dcm' -type f -printf '%s\n' 2>/dev/null | \
        awk '{s+=$1} END {print s+0}')
    disk_human=$(numfmt --to=iec-i --suffix=B "$disk_bytes" 2>/dev/null || echo "${disk_bytes}B")
    info_csv="$dataset_dir/dataset_info.csv"
    if [[ ! -f "$info_csv" ]]; then
        printf 'dataset,disk_bytes,disk_human\n%s,%s,%s\n' "$dataset_clean" "$disk_bytes" "$disk_human" > "$info_csv"
    fi

    for strategy_entry in "${STRATEGIES[@]}"; do
        IFS='|' read -r strat_label strat_args strat_file <<< "$strategy_entry"

        csv_path="$dataset_dir/${strat_file}.csv"
        ppm_path="$dataset_dir/${strat_file}.ppm"

        run_count=$((run_count + 1))

        # --resume: skip if CSV already exists and has content
        if [[ "$RESUME" == true ]] && [[ -s "$csv_path" ]]; then
            echo "[$run_count/$total_runs] SKIP (exists): $dataset_clean / $strat_label"
            skip_count=$((skip_count + 1))
            continue
        fi

        cmd=("$BINARY" "$series_path"
             --mode $strat_args
             --frames "$FRAMES"
             --metrics "$csv_path"
             --snapshot "$ppm_path"
             --summary "$OUTPUT_ROOT/results.csv"
             --dataset "$dataset_clean"
             --strategy "$strat_label"
             --disk-bytes "$disk_bytes"
             --build-time)

        echo "[$run_count/$total_runs] $dataset_clean / $strat_label"

        if [[ "$DRY_RUN" == true ]]; then
            echo "  CMD: ${cmd[*]}"
            continue
        fi

        if "${cmd[@]}" > "$dataset_dir/${strat_file}.log" 2>&1; then
            echo "  OK  → $csv_path"
            # Capture the rendering-relevant sparsity % from this run's log and
            # record it (with the epsilon used) into the dataset sparsity manifest.
            if grep -q 'rendering-relevant' "$dataset_dir/${strat_file}.log"; then
                eps=$(grep -oE 'eps=[0-9.]+' "$dataset_dir/${strat_file}.log" | head -n1 | cut -d= -f2)
                pct=$(grep -oE '[0-9.]+% sparse' "$dataset_dir/${strat_file}.log" | head -n1 | sed 's/% sparse//')
                sparsity_csv="$dataset_dir/sparsity.csv"
                if [[ ! -f "$sparsity_csv" ]]; then
                    printf 'strategy,epsilon,sparsity_pct\n' > "$sparsity_csv"
                fi
                printf '%s,%s,%s\n' "$strat_label" "${eps:-}" "$pct" >> "$sparsity_csv"
            fi
        else
            echo "  FAIL (exit $?) → see $dataset_dir/${strat_file}.log"
            fail_count=$((fail_count + 1))
            # Write a marker so --resume skips it next time
            echo "FAILED" > "$csv_path"
        fi
    done
done

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo " Test run complete"
echo "=============================="
echo " Total runs  : $total_runs"
echo " Executed    : $((run_count - skip_count - fail_count))"
echo " Skipped     : $skip_count"
echo " Failed      : $fail_count"
echo " Results dir : $OUTPUT_ROOT"
echo " Log file    : $LOGFILE"