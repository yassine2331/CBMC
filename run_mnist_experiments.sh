#!/bin/bash
# ============================================================
#  SLURM job: MNIST experiments — CNN | CBM | CEM
#  Submit with:  sbatch run_mnist_experiments.sh
#  Local run:    bash run_mnist_experiments.sh
# ============================================================

# ── SLURM resource directives ────────────────────────────────
#SBATCH --job-name=cbmc_mnist
#SBATCH --output=logs/cbmc_mnist_%j.out
#SBATCH --error=logs/cbmc_mnist_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# ============================================================
#  USER VARIABLES — edit these before submitting
# ============================================================

EPOCHS=20              # training epochs per experiment
DEVICE="mps"          # cuda | mps | cpu
OPERATORS="+ x"        # arithmetic operators to use: + - x /

DIGITS_SINGLE="9"              # condition 1: single digit
DIGITS_ALL="1 2 3 4 5 6 7 8 9" # condition 2: all digits

# ============================================================
#  Environment setup
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"   # always run from repo root

# Load cluster modules if available (adjust names to your cluster)
if command -v module &>/dev/null; then
    module load cuda/12.1
    module load python/3.11
fi

# Uncomment to activate your environment:
# source venv/bin/activate
# conda activate cbmc

mkdir -p logs outputs/results

echo "============================================================"
echo " Job:     ${SLURM_JOB_ID:-local}"
echo " Node:    $(hostname)"
echo " Device:  $DEVICE  |  Epochs: $EPOCHS"
echo " Ops:     $OPERATORS"
echo "============================================================"

# ── helper: run one experiment and save directly to a tagged file ──
run_exp () {
    local name="$1"       # experiment key, e.g. exp_cls_mnist
    local tag="$2"        # output tag,     e.g. digit9 / all_digits
    local digits="$3"     # digit list,     e.g. "9" or "1 2 3 4 5 6 7 8 9"

    echo ""
    echo "── $name  [$tag] ──────────────────────────────────────"

    # --tag tells Python to write outputs/results/<name>_<tag>.csv directly —
    # no renaming needed and no risk of a later run overwriting an earlier one.
    python scripts/run_experiment.py \
        --exp      "$name"           \
        --epochs   "$EPOCHS"         \
        --device   "$DEVICE"         \
        --operators $OPERATORS       \
        --digits   $digits           \
        --tag      "$tag"

    echo "  -> saved to outputs/results/${name}_${tag}.csv"
}

# ============================================================
#  Experiments
# ============================================================

# ── 1. CNN baseline (no concepts) ────────────────────────────
run_exp  exp_cls_mnist  digit9       "$DIGITS_SINGLE"
run_exp  exp_cls_mnist  all_digits   "$DIGITS_ALL"

# ── 2. CBM (Concept Bottleneck Model) ────────────────────────
run_exp  exp_cbm_cls_mnist  digit9       "$DIGITS_SINGLE"
run_exp  exp_cbm_cls_mnist  all_digits   "$DIGITS_ALL"

# ── 3. CEM (Concept Embedding Model) ─────────────────────────
run_exp  exp_cem_cls_mnist  digit9       "$DIGITS_SINGLE"
run_exp  exp_cem_cls_mnist  all_digits   "$DIGITS_ALL"

# ============================================================
#  Summary
# ============================================================

echo ""
echo "============================================================"
echo " Done. Final CSVs in outputs/results/:"
echo "============================================================"
ls outputs/results/*.csv 2>/dev/null
