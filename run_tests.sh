#!/usr/bin/env bash
#
# run_tests.sh — Run all rendering strategies with varied parameters
# across every TCIA dataset found in ~/Documents/TCIA.
#
# Results (CSV metrics + PPM snapshots) are stored in ../optix_test_results/
# relative to the project root (i.e. beside this repo as a brother folder).
#
# Usage:
#   ./run_tests.sh              # run all tests
#   ./run_tests.sh --dry-run    # print commands without executing
#   ./run_tests.sh --resume     # skip combinations that already have a CSV
#

set -euo pipefail

# ── resolve project root (where this script lives) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BINARY="$PROJECT_ROOT/build/bin/optix_app"

# ── output root (brother folder of the project) ─────────────────────────────
OUTPUT_ROOT="$(dirname "$PROJECT_ROOT")/optix_test_results"

# ── TCIA data root ──────────────────────────────────────────────────────────
TCIA_ROOT="${TCIA_ROOT:-$HOME/Documents/TCIA}"

# ── CLI flags ───────────────────────────────────────────────────────────────
DRY_RUN=false
RESUME=false
FRAMES=200

for arg in "$@"; do
    case "$arg" in
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
echo ""

# ── strategy configurations ─────────────────────────────────────────────────
# Each entry is a label followed by the CLI args for that configuration.
# The label is used for directory/file naming (must be filesystem-safe).

STRATEGIES=()

# --- modes with no extra params ---
STRATEGIES+=("manual|manual|manual")
STRATEGIES+=("optix|optix|optix")

# --- bricked-regions: vary brick-size (default epsilon) ---
STRATEGIES+=("bricked-regions_brick8|bricked-regions --brick-size 8|bricked-regions--brick8")
STRATEGIES+=("bricked-regions_brick32|bricked-regions --brick-size 32|bricked-regions--brick32")

# --- octree: vary leaf-size (default epsilon) ---
STRATEGIES+=("octree_leaf4|octree --leaf-size 4|octree--leaf4")
STRATEGIES+=("octree_leaf8_adapt|octree --leaf-size 8 --adaptive-march on|octree--leaf8-adapt")

# --- octree-regions: vary leaf-size (default epsilon) ---
STRATEGIES+=("octree-regions_leaf4|octree-regions --leaf-size 4|octree-regions--leaf4")
STRATEGIES+=("octree-regions_leaf8_adapt|octree-regions --leaf-size 8 --adaptive-march on|octree-regions--leaf8-adapt")

# --- nanovdb: vary sampler (default epsilon) ---
STRATEGIES+=("nanovdb_trilinear|nanovdb --nanovdb-sampler trilinear|nanovdb--trilinear")

# --- epsilon sweep ---
# Sweep epsilon across the strategies that consume it:
#   adaptive, bricked-regions (brick16), octree (leaf8),
#   octree-regions (leaf8), nanovdb (nearest)
EPSILONS=(0.01 0.05 0.1 0.2 0.5)

for eps in "${EPSILONS[@]}"; do
    eps_tag=$(printf "eps%03d" "$(awk -v e="$eps" 'BEGIN { printf "%d", e*100 }')")
    STRATEGIES+=("adaptive_$eps_tag|adaptive --epsilon $eps|adaptive--$eps_tag")
    STRATEGIES+=("bricked-regions_brick16_$eps_tag|bricked-regions --brick-size 16 --epsilon $eps|bricked-regions--brick16-$eps_tag")
    STRATEGIES+=("octree_leaf8_$eps_tag|octree --leaf-size 8 --epsilon $eps|octree--leaf8-$eps_tag")
    STRATEGIES+=("octree-regions_leaf8_$eps_tag|octree-regions --leaf-size 8 --epsilon $eps|octree-regions--leaf8-$eps_tag")
    STRATEGIES+=("nanovdb_nearest_$eps_tag|nanovdb --nanovdb-sampler nearest --epsilon $eps|nanovdb--nearest-$eps_tag")
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

    for strategy_entry in "${STRATEGIES[@]}"; do
        IFS='|' read -r strat_label strat_args strat_file <<< "$strategy_entry"

        csv_path="$dataset_dir/${strat_file}.csv"
        ppm_path="$dataset_dir/${strat_file}.ppm"
        series_path="$TCIA_ROOT/$dataset"

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
             --snapshot "$ppm_path")

        echo "[$run_count/$total_runs] $dataset_clean / $strat_label"

        if [[ "$DRY_RUN" == true ]]; then
            echo "  CMD: ${cmd[*]}"
            continue
        fi

        if "${cmd[@]}" >> "$dataset_dir/${strat_file}.log" 2>&1; then
            echo "  OK  → $csv_path"
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
