#!/usr/bin/env bash
# ============================================================
# Image fidelity pipeline: compares every strategy's rendered
# snapshot against its dataset's manual reference image, then
# plots the results. Run from anywhere:
#
#     ./run_image_fidelity.sh
#
# Steps:
#   D1  compute PSNR/SSIM/MAE (grayscale + per-channel RGB)
#   D2  fidelity by family at epsilon = 0.01
#   D3  fidelity vs epsilon sweep per family
#   D4  difference maps for one representative dataset
#   D5  fidelity vs skip-ratio and vs speedup (Pareto)
#   D6  fidelity per STRATEGY x epsilon (luminance & RGB figures)
#   D7  image table: rows = strategies, columns = epsilon
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"
PY="venv/bin/python"

banner() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
}

banner "D1 - computing fidelity metrics (slowest step, progress bar below)"
"$PY" d1_image_fidelity.py

banner "D2 - fidelity by family (epsilon = 0.01)"
"$PY" d2_fidelity_by_family.py

banner "D3 - fidelity vs epsilon"
"$PY" d3_fidelity_vs_epsilon.py

banner "D4 - difference maps"
"$PY" d4_difference_maps.py

banner "D5 - fidelity vs speed / skip"
"$PY" d5_fidelity_vs_speed.py

banner "D6 - fidelity per strategy & epsilon (luminance + RGB)"
"$PY" d6_fidelity_per_epsilon.py

banner "D7 - image grid (first dataset). Pass a dataset name via: $PY d7_image_grid.py <series>"
"$PY" d7_image_grid.py

banner "D8 - image grid, epsilon-sweep strategies only"
"$PY" d8_image_grid_sweep.py

echo ""
echo "Done. Plots are in: ./plots/"
echo "Per-row metrics:   /home/adri/projects/TFM/optix_test_results/image_fidelity_metrics.csv"