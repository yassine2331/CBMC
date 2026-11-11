#!/bin/bash
# =============================================================================
#  run_benchmark.sh — every concept representation on every dataset, N seeds
# =============================================================================
#
#  Wraps scripts/run_benchmark.py. Same encoder, head and training loop for all
#  models; only the bottleneck differs:
#
#    cbm  cem (ours, Case 3.5)  cem_linear  cem_linear_raw  binned5  binned10  binned15
#    (+ nn, no bottleneck, if you ask for it with --models)
#
#  Datasets: mnist  pendulum  lidc        (oai: not runnable until images exist)
#
#  Metrics per seed: task, intervention (all concepts true), and the ZERO test:
#  on test samples with a concept near 0, fix only that concept to the truth.
#
#
#  HOW TO RUN
#  ----------
#    bash   run_benchmark.sh --parallel                       # ONE JOB PER dataset x model
#                                                             #   (3 x 7 = 21 jobs), login node
#    bash   run_benchmark.sh --parallel --datasets lidc       # 7 jobs, LIDC only
#    sbatch run_benchmark.sh --datasets mnist                 # one job, all mnist models in turn
#    sbatch run_benchmark.sh --datasets mnist --models "cem cem_linear" --seeds 10
#    bash   run_benchmark.sh --summarize                      # mean ± std tables, no training
#
#  --parallel must be run with bash on the LOGIN node (it calls sbatch itself).
#
#
#  OPTIONS (defaults in brackets)
#  -------
#    --datasets "..."      [mnist pendulum lidc]
#    --models "..."        [cbm cem cem_linear cem_linear_raw binned5 binned10 binned15]
#    --seeds N             [30]  seeds 41 .. 41+N-1
#    --epochs N            [per dataset: mnist 30, pendulum 20, lidc 250]
#    --parallel            one sbatch job per dataset x model
#    --summarize           only aggregate existing results
#    --extra "..."         passed straight to run_benchmark.py, e.g.
#                          --extra "--concept-weight 5 --max-train 20000"
#
#
#  RESUMING
#  --------
#  Each finished seed is appended to outputs/results/benchmark/<dataset>/<model>.csv
#  immediately, and a rerun skips seeds that are already there. If a job hits
#  the 24 h wall time, submit the same command again.
#
#
#  COST (rough, per model job of 30 seeds)
#  ----
#    mnist     ~2-3 min/seed   -> ~1-1.5 h
#    pendulum  ~15-20 min/seed -> ~8-10 h   (use --extra "--max-train 20000" to cut it ~3x)
#    lidc      ~13 min/seed    -> ~6.5 h    (250 epochs x ~3 s, as in the paper runs)
#
#
#  OUTPUT, and getting it back to the laptop
#  -----------------------------------------
#    outputs/results/benchmark/<dataset>/<model>.csv      one row per seed
#    outputs/results/benchmark/<dataset>_summary.csv      mean / std
#    outputs/models/benchmark/<dataset>/<model>_seed<N>.pt
#
#      rsync -av 'ouesly@piz:/home/ouesly/CBMC/outputs/results/benchmark' outputs/results/
#
# =============================================================================

#SBATCH --job-name=cbmc-bench
#SBATCH --time=24:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --container-image=mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail          # NOT -e: a failing model must not abort the rest

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
MAMBA_CMD="${MAMBA_CMD:-mamba run -p ./env}"   # override to test locally

usage () { sed -n '2,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; }

DATASETS="mnist pendulum lidc"
MODELS="cbm cem cem_linear cem_linear_raw binned5 binned10 binned15"
SEEDS=30
EPOCHS=""
PARALLEL=""
SUMMARIZE=""
EXTRA=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)    usage; exit 0 ;;
        --datasets)   DATASETS="$2"; shift 2 ;;
        --models)     MODELS="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --extra)      EXTRA="$2"; shift 2 ;;
        --parallel)   PARALLEL=1; shift ;;
        --summarize)  SUMMARIZE=1; shift ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
done

for d in $DATASETS; do
    case "$d" in mnist|pendulum|lidc) ;;
        oai) echo "ERROR: oai has no images yet; see _oai() in scripts/run_benchmark.py" >&2; exit 1 ;;
        *)   echo "ERROR: unknown dataset '$d' (mnist pendulum lidc)" >&2; exit 1 ;;
    esac
done
for m in $MODELS; do
    case "$m" in cbm|cem|cem_linear|cem_linear_raw|binned5|binned10|binned15|nn) ;;
        *) echo "ERROR: unknown model '$m'" >&2; exit 1 ;;
    esac
done

if [ -n "$SUMMARIZE" ]; then
    for d in $DATASETS; do
        $MAMBA_CMD python scripts/run_benchmark.py --dataset "$d" --summarize
    done
    exit 0
fi

# --- parallel: one job per dataset x model ------------------------------------
if [ -n "$PARALLEL" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "ERROR: --parallel submits jobs itself; run it on the login node with" >&2
        echo "       bash run_benchmark.sh ... --parallel   (not sbatch)" >&2
        exit 1
    fi
    mkdir -p logs
    n=0
    for d in $DATASETS; do
        for m in $MODELS; do
            args=(--datasets "$d" --models "$m" --seeds "$SEEDS")
            [ -n "$EPOCHS" ] && args+=(--epochs "$EPOCHS")
            [ -n "$EXTRA" ]  && args+=(--extra "$EXTRA")
            echo "+ sbatch --job-name=bench-$d-$m $0 ${args[*]}"
            sbatch --job-name="bench-$d-$m" "$0" "${args[@]}"
            n=$((n + 1))
        done
    done
    echo "submitted $n jobs"
    exit 0
fi

# --- sequential, inside one job ----------------------------------------------
export PYTHONUNBUFFERED=1
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}   Host: $(hostname)   Start: $(date)"
echo "Datasets:  $DATASETS"
echo "Models:    $MODELS"
echo "Seeds:     $SEEDS   Epochs: ${EPOCHS:-dataset default}   Extra: ${EXTRA:-none}"
echo "========================================"

FAILED=""
for d in $DATASETS; do
    CMD="python scripts/run_benchmark.py --dataset $d --models $MODELS --seeds $SEEDS --num-workers 4"
    [ -n "$EPOCHS" ] && CMD="$CMD --epochs $EPOCHS"
    [ -n "$EXTRA" ]  && CMD="$CMD $EXTRA"
    echo ""
    echo "+ $CMD"
    $MAMBA_CMD $CMD || { echo "$d FAILED" >&2; FAILED="$FAILED $d"; }
done

echo ""
echo "End: $(date)"
[ -n "$FAILED" ] && { echo "FAILED:$FAILED"; exit 1; }
exit 0
