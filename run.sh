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
MAMBA_CMD="mamba run -p ./env"

EXP_TYPE="${1:-ALL}"

if [ "$EXP_TYPE" == "ALL" ]; then
    echo "Running all 4 experiments sequentially via mamba run..."

    $MAMBA_CMD python scripts/run_experiment.py --exp exp_gen_mnist
    $MAMBA_CMD python scripts/run_experiment.py --exp exp_gen_pendulum
    $MAMBA_CMD python scripts/run_experiment.py --exp exp_cls_mnist
    $MAMBA_CMD python scripts/run_experiment.py --exp exp_cls_pendulum
else
    echo "Running specific experiment: $EXP_TYPE"
    $MAMBA_CMD python scripts/run_experiment.py --exp "$EXP_TYPE"
fi

# --- 4. Cleanup ---
STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"

exit $STATUS
