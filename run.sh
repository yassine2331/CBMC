#!/bin/bash
#SBATCH --job-name=cbmc-experiments
#SBATCH --time=24:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --container-image=mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- 1. Path Setup ---

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

# --- 2. Logging Info ---
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "Directory: $PROJECT_DIR"
echo "========================================"

# --- 3. Execution using mamba run ---
#
# Usage:
#   sbatch run.sh                          # all experiments, 5 seeds each
#   sbatch run.sh exp_cem_cls_mnist        # one experiment,  5 seeds
#   sbatch run.sh exp_cem_cls_mnist 10     # one experiment, 10 seeds
#   sbatch run.sh ALL 10                   # all experiments, 10 seeds
#   RUNS=10 sbatch run.sh                  # same, via environment variable
#
MAMBA_CMD="mamba run -p ./env"

EXP_TYPE="${1:-ALL}"
# Number of seeds per experiment. Each run uses seed 41+i; the mean and std
# across these runs is what lands in outputs/results/<exp>_stats.csv.
# Must be > 1 — with --runs 1 the runner reports metrics but saves no stats file.
RUNS="${2:-${RUNS:-5}}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -lt 2 ]; then
    echo "ERROR: runs must be an integer >= 2 (got '$RUNS')" >&2
    exit 1
fi

echo "Runs per experiment: $RUNS"

if [ "$EXP_TYPE" == "ALL" ]; then
    echo "Running all 4 experiments sequentially via mamba run..."

    for exp in exp_gen_mnist exp_gen_pendulum exp_cls_mnist exp_cls_pendulum; do
        $MAMBA_CMD python scripts/run_experiment.py --exp "$exp" --runs "$RUNS"
    done
else
    echo "Running specific experiment: $EXP_TYPE"
    $MAMBA_CMD python scripts/run_experiment.py --exp "$EXP_TYPE" --runs "$RUNS"
fi

# --- 4. Aggregate every *_stats.csv into one summary table ---
# Writes outputs/results/summary_all_experiments{,_pretty}.csv
$MAMBA_CMD python scripts/aggregate_results.py

# --- 5. Cleanup ---
STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"

exit $STATUS
