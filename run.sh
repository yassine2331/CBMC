#!/bin/bash
#SBATCH --job-name=cbmc-experiments
#SBATCH --time=24:00:00            
#SBATCH --gres=gpu:1               
#SBATCH --mem=32G                  
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- 1. Path Setup ---
PROJECT_DIR="$(pwd)" 
cd "$PROJECT_DIR"

# --- 2. Logging Info ---
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "Directory: $PROJECT_DIR"
echo "========================================"

# --- 3. Execution using mamba run ---
# You can use your exact path (-p /home/username/my_project/env) 
# or the named environment (-n cbmc). I'll use the named one here.
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
