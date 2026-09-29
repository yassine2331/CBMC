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

usage () {
    cat <<EOF
Usage: sbatch run.sh [EXPERIMENT|ALL] [--runs N]

  EXPERIMENT   experiment key, e.g. exp_cem_cls_mnist. Omit or pass ALL
               to run exp_gen_mnist, exp_gen_pendulum, exp_cls_mnist and
               exp_cls_pendulum in sequence.
  --runs N     number of seeds per experiment (default 5, minimum 2).
               May also be given positionally, or as the RUNS env var.

Examples:
  sbatch run.sh                              # everything, 5 seeds
  sbatch run.sh --runs 30                    # everything, 30 seeds
  sbatch run.sh exp_cem_cls_mnist            # one experiment, 5 seeds
  sbatch run.sh exp_cem_cls_mnist --runs 30  # one experiment, 30 seeds
  sbatch run.sh exp_cem_cls_mnist 30         # same, positional form
EOF
}

EXP_TYPE=""
RUNS="${RUNS:-5}"          # env var default, overridden by the flag below

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage; exit 0 ;;
        -r|--runs)
            if [ $# -lt 2 ]; then
                echo "ERROR: --runs needs a number after it." >&2; usage >&2; exit 1
            fi
            RUNS="$2"; shift 2 ;;
        --runs=*)
            RUNS="${1#*=}"; shift ;;
        -*)
            echo "ERROR: unknown option '$1'." >&2; usage >&2; exit 1 ;;
        *)
            if [ -z "$EXP_TYPE" ]; then
                EXP_TYPE="$1"
            elif [[ "$1" =~ ^[0-9]+$ ]]; then
                RUNS="$1"                     # positional run count
            else
                echo "ERROR: unexpected argument '$1'." >&2; usage >&2; exit 1
            fi
            shift ;;
    esac
done

EXP_TYPE="${EXP_TYPE:-ALL}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -lt 2 ]; then
    echo "ERROR: runs must be an integer >= 2 (got '$RUNS')." >&2
    usage >&2
    exit 1
fi

echo "Experiment:          $EXP_TYPE"
echo "Runs per experiment: $RUNS"

# Run one experiment, echoing the exact command first so the .out log always
# records what was actually executed.
run_one () {
    echo ""
    echo "+ python scripts/run_experiment.py --exp $1 --runs $RUNS"
    $MAMBA_CMD python scripts/run_experiment.py --exp "$1" --runs "$RUNS"
}

if [ "$EXP_TYPE" == "ALL" ]; then
    echo "Running all 4 experiments sequentially via mamba run..."

    for exp in exp_gen_mnist exp_gen_pendulum exp_cls_mnist exp_cls_pendulum; do
        run_one "$exp"
    done
else
    echo "Running specific experiment: $EXP_TYPE"
    run_one "$EXP_TYPE"
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
