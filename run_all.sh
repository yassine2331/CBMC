#!/bin/bash
# =============================================================================
#  run_all.sh — run every experiment, one after another, from one submission
# =============================================================================
#
#  Submits ONE job that runs each experiment script in sequence. Nothing is
#  backgrounded, so each finishes completely before the next starts, and they
#  never compete for the GPU.
#
#  ORDER, and why
#  --------------
#    1  overfit   test_overfit.sh    Diagnostic. Trains the plain CNN with and
#                                    without regularisation. RUN THIS FIRST:
#                                    if regularisation closes the train/test
#                                    gap, every later comparison should be
#                                    re-run with it, so knowing the answer
#                                    early saves repeating stage 2.
#    2  lidc      run_lidc_all.sh    The seven-variant concept comparison.
#    3  binned    run_binned.sh      ArithmeticMNIST binning sweep (Case 3 vs
#                                    Case 3.5). Independent of the LIDC data.
#
#  HOW TO RUN
#  ----------
#    sbatch run_all.sh                          # all three, defaults
#    sbatch run_all.sh --only "overfit lidc"    # a subset, in that order
#    sbatch run_all.sh --runs 10 --epochs 60    # passed to every stage
#    sbatch run_all.sh --augment                # passed to the stages that take it
#    bash   run_all.sh --help
#
#  Env vars: RUNS, EPOCHS, AUGMENT, WEIGHT_DECAY, BACKBONE_DROPOUT, SUFFIX,
#  DEVICE, ONLY.
#
#  REGULARISATION — pass these or the models memorise the training set:
#    --augment --weight-decay 0.01 --backbone-dropout 0.1
#  Measured on the plain CNN over 10 seeds: train/test gap 0.234 -> -0.047,
#  test balanced accuracy 0.763 -> 0.819, seed std 0.081 -> 0.027.
#
#  A stage that fails does NOT stop the others: each runs in its own subshell
#  and its exit status is recorded, so one crash late at night does not cost
#  you the whole queue slot. The summary at the end lists what passed.
#
#  COST
#  ----
#  With defaults this is several hours and may exceed the 24 h wall time.
#  Check the per-stage estimates printed at the start, and use --only or
#  lower --runs if the total looks too close to the limit.
#
# =============================================================================

#SBATCH --job-name=cbmc-all
#SBATCH --time=24:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --container-image=mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail          # NOT -e: a failing stage must not abort the rest
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

usage () { sed -n '2,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; }

RUNS="${RUNS:-}"
EPOCHS="${EPOCHS:-}"
AUGMENT="${AUGMENT:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
BACKBONE_DROPOUT="${BACKBONE_DROPOUT:-}"
SUFFIX="${SUFFIX:-}"
DEVICE="${DEVICE:-}"
ONLY="${ONLY:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --runs)     RUNS="$2"; shift 2 ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        --augment)            AUGMENT=1; shift ;;
        --weight-decay)       WEIGHT_DECAY="$2"; shift 2 ;;
        --backbone-dropout)   BACKBONE_DROPOUT="$2"; shift 2 ;;
        --suffix)   SUFFIX="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --only)     ONLY="$2"; shift 2 ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
done

ALL="overfit lidc binned"
STAGES="${ONLY:-$ALL}"
for st in $STAGES; do
    case "$st" in
        overfit|lidc|binned) ;;
        *) echo "ERROR: unknown stage '$st'. Choose from: $ALL" >&2; exit 1 ;;
    esac
done

# shared flags, only those each stage actually accepts
COMMON=""
[ -n "$RUNS" ]   && COMMON="$COMMON --runs $RUNS"
[ -n "$EPOCHS" ] && COMMON="$COMMON --epochs $EPOCHS"
[ -n "$DEVICE" ] && COMMON="$COMMON --device $DEVICE"
[ -n "$SUFFIX" ] && COMMON="$COMMON --suffix $SUFFIX"

stage_cmd () {
    case "$1" in
        overfit) echo "bash test_overfit.sh $COMMON"\
                      "${WEIGHT_DECAY:+--weight-decay $WEIGHT_DECAY}"\
                      "${BACKBONE_DROPOUT:+--dropout $BACKBONE_DROPOUT}" ;;
        lidc)    echo "bash run_lidc_all.sh $COMMON ${AUGMENT:+--augment}"\
                      "${WEIGHT_DECAY:+--weight-decay $WEIGHT_DECAY}"\
                      "${BACKBONE_DROPOUT:+--backbone-dropout $BACKBONE_DROPOUT}" ;;
        binned)  echo "bash run_binned.sh ${RUNS:+--runs $RUNS} ${EPOCHS:+--epochs $EPOCHS} ${SUFFIX:+--tag $SUFFIX}" ;;
    esac
}

echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "Stages:    $STAGES"
echo "========================================"
echo ""
echo "Planned, in order:"
i=0
for st in $STAGES; do i=$((i+1)); echo "  $i. $st   ->  $(stage_cmd "$st")"; done
echo ""

N=$(echo $STAGES | wc -w | tr -d ' ')
i=0
RESULTS=""
JOB_T0=$(date +%s)

for st in $STAGES; do
    i=$((i+1))
    T0=$(date +%s)
    echo ""
    echo "############################################################"
    echo "# STAGE $i/$N — $st        started $(date +%H:%M:%S)"
    echo "############################################################"
    echo "+ $(stage_cmd "$st")"
    echo ""

    # Subshell, so a stage calling `exit` cannot kill this script.
    ( eval "$(stage_cmd "$st")" )
    rc=$?

    MINS=$(( ($(date +%s) - T0) / 60 ))
    if [ $rc -eq 0 ]; then
        echo ""
        echo "# STAGE $i/$N — $st OK (${MINS} min)"
        RESULTS="$RESULTS\n  $st: OK (${MINS} min)"
    else
        echo ""
        echo "# STAGE $i/$N — $st FAILED (exit $rc, ${MINS} min) — continuing" >&2
        RESULTS="$RESULTS\n  $st: FAILED exit $rc (${MINS} min)"
    fi
done

echo ""
echo "========================================"
echo "All stages finished after $(( ($(date +%s) - JOB_T0) / 60 )) min"
printf "$RESULTS\n"
echo ""
echo "Results in outputs/results/, training curves in outputs/logs/"
echo "End: $(date)"
echo "========================================"
