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

  EXPERIMENT   an experiment key (e.g. exp_cem_cls_mnist) or a group:
                 ALL        every experiment below (14)  [default]
                 BASELINES  the 4 no-concept baselines
                 CBM        the 4 CBM experiments
                 CEM        the 4 CEM experiments + 2 ablations
                 CLS        every classification/regression experiment
                 GEN        every generation (VAE) experiment
  --runs N     number of seeds per experiment (default 5, minimum 2).
               May also be given positionally, or as the RUNS env var.
  --epochs N   training epochs, overriding each experiment's JSON config.
               Omit to use the per-experiment config value. Also EPOCHS env var.

Examples:
  sbatch run.sh                              # everything, 5 seeds
  sbatch run.sh --runs 30                    # everything, 30 seeds
  sbatch run.sh exp_cem_cls_mnist            # one experiment, 5 seeds
  sbatch run.sh exp_cem_cls_mnist --runs 30  # one experiment, 30 seeds
  sbatch run.sh exp_cem_cls_mnist 30         # same, positional form
  sbatch run.sh CEM --runs 5 --epochs 50     # 50 epochs instead of the config's
EOF
}

EXP_TYPE=""
RUNS="${RUNS:-5}"          # env var default, overridden by the flag below
EPOCHS="${EPOCHS:-}"       # empty = use each experiment's own config value

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
        -e|--epochs)
            if [ $# -lt 2 ]; then
                echo "ERROR: --epochs needs a number after it." >&2; usage >&2; exit 1
            fi
            EPOCHS="$2"; shift 2 ;;
        --epochs=*)
            EPOCHS="${1#*=}"; shift ;;
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

if [ -n "$EPOCHS" ] && { ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; }; then
    echo "ERROR: epochs must be an integer >= 1 (got '$EPOCHS')." >&2
    usage >&2
    exit 1
fi

# Passed through to run_experiment.py only when set, so an unset value leaves
# each experiment's JSON config in charge.
EPOCH_ARG=""
[ -n "$EPOCHS" ] && EPOCH_ARG="--epochs $EPOCHS"

echo "Experiment:          $EXP_TYPE"
echo "Runs per experiment: $RUNS"
echo "Epochs:              ${EPOCHS:-<from each config>}"

# Run one experiment, echoing the exact command first so the .out log always
# records what was actually executed.
run_one () {
    echo ""
    echo "+ python scripts/run_experiment.py --exp $1 --runs $RUNS $EPOCH_ARG"
    $MAMBA_CMD python scripts/run_experiment.py --exp "$1" --runs "$RUNS" $EPOCH_ARG
}

# Experiment groups. "ALL" means every model, so the summary table gets a
# column per model — running only the baselines gives a one-column table.
BASELINES="exp_gen_mnist exp_gen_pendulum exp_cls_mnist exp_cls_pendulum"
CBM_EXPS="exp_cbm_gen_mnist exp_cbm_gen_pendulum exp_cbm_cls_mnist exp_cbm_cls_pendulum"
CEM_EXPS="exp_cem_gen_mnist exp_cem_gen_pendulum exp_cem_cls_mnist exp_cem_cls_pendulum"
ABLATIONS="exp_cem_tanh_cls_mnist exp_cem_linear_cls_mnist"

case "$EXP_TYPE" in
    ALL)        GROUP="$BASELINES $CBM_EXPS $CEM_EXPS $ABLATIONS" ;;
    BASELINES)  GROUP="$BASELINES" ;;
    CBM)        GROUP="$CBM_EXPS" ;;
    CEM)        GROUP="$CEM_EXPS $ABLATIONS" ;;
    CLS)        GROUP="exp_cls_mnist exp_cls_pendulum exp_cbm_cls_mnist exp_cbm_cls_pendulum exp_cem_cls_mnist exp_cem_cls_pendulum $ABLATIONS" ;;
    GEN)        GROUP="exp_gen_mnist exp_gen_pendulum exp_cbm_gen_mnist exp_cbm_gen_pendulum exp_cem_gen_mnist exp_cem_gen_pendulum" ;;
    *)          GROUP="$EXP_TYPE" ;;
esac

N_EXPS=$(echo $GROUP | wc -w | tr -d ' ')
echo "Experiments to run:  $N_EXPS  ($N_EXPS x $RUNS = $((N_EXPS * RUNS)) trainings)"

for exp in $GROUP; do
    run_one "$exp"
done

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
