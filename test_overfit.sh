#!/bin/bash
# =============================================================================
#  test_overfit.sh — does regularisation close the train/test gap?
# =============================================================================
#
#  WHY THIS EXISTS
#  ---------------
#  The LIDC comparison ran with essentially no regularisation: no augmentation,
#  no weight decay, encoder dropout 0.0, no early stopping. Every model reached
#  0.96-1.00 training accuracy against 0.74-0.79 test balanced accuracy -- a
#  ~20 point gap on 944 training volumes. With every model memorising, the
#  ranking between concept representations is not trustworthy.
#
#  This runs ONE model (the plain CNN, no bottleneck) twice, changing only the
#  regularisation, so the effect is isolated from any concept machinery:
#
#    A  plain       no augmentation, no weight decay, dropout 0.0   (as before)
#    B  regularised augmentation + weight decay + encoder dropout
#
#  Read the TRAIN/TEST GAP, not the test accuracy. If B's gap is much smaller,
#  regularisation is the fix and the full comparison should be re-run with it.
#  If both gaps stay large, the problem is elsewhere -- model capacity or
#  dataset size -- and early stopping or a smaller encoder is the next lever.
#
#
#  HOW TO RUN
#  ----------
#    sbatch test_overfit.sh                       # defaults below
#    sbatch test_overfit.sh --runs 10 --epochs 60
#    sbatch test_overfit.sh --bottleneck cem      # repeat with the bottleneck
#    bash   test_overfit.sh --help
#
#  Env vars: RUNS, EPOCHS, BOTTLENECK, WEIGHT_DECAY, DROPOUT, DEVICE, SUFFIX.
#
#
#  WHAT IS VARIED
#  --------------
#    --weight-decay W   AdamW weight decay for arm B           default 0.01
#    --dropout D        Dropout3d after each encoder block, B  default 0.1
#    --bottleneck NAME  which model to test                    default none
#                       ('none' = plain CNN, the cleanest test)
#    --runs N           seeds per arm                          default 10
#    --epochs N         epochs per run                         default 100
#
#  Augmentation in arm B is all 48 cube symmetries plus +/-3 voxel jitter.
#  These are ISOMETRIES only: three of the concepts are diameter, volume and
#  surface area, so any rescaling or deformation would corrupt their labels.
#
#
#  OUTPUT
#  ------
#    outputs/results/lidc_overfit_plain<suffix>.csv
#    outputs/results/lidc_overfit_reg<suffix>.csv
#    outputs/logs/lidc_overfit_*_history.csv     per-epoch curves
#
#  The script prints a train/test gap comparison at the end.
#
# =============================================================================

#SBATCH --job-name=cbmc-overfit
#SBATCH --time=24:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --container-image=mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "========================================"

MAMBA_CMD="mamba run -p ./env"
usage () { sed -n '2,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; }

RUNS="${RUNS:-10}"
EPOCHS="${EPOCHS:-100}"
BOTTLENECK="${BOTTLENECK:-none}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DROPOUT="${DROPOUT:-0.1}"
SUFFIX="${SUFFIX:-}"
DEVICE="${DEVICE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --runs)           RUNS="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --bottleneck)     BOTTLENECK="$2"; shift 2 ;;
        --weight-decay)   WEIGHT_DECAY="$2"; shift 2 ;;
        --dropout)        DROPOUT="$2"; shift 2 ;;
        --suffix)         SUFFIX="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
done

if [ ! -f data/processed/lidc/nodules.csv ]; then
    echo "ERROR: data/processed/lidc/nodules.csv not found." >&2
    echo "Build the cubes first:  python scripts/prepare_lidc.py --all" >&2
    exit 1
fi

COMMON="--bottleneck $BOTTLENECK --runs $RUNS --epochs $EPOCHS"
[ -n "$DEVICE" ] && COMMON="$COMMON --device $DEVICE"

echo "Model:          $BOTTLENECK"
echo "Seeds per arm:  $RUNS      Epochs: $EPOCHS"
echo "Arm A (plain):  no augmentation, no weight decay, dropout 0.0"
echo "Arm B (reg):    augment + weight decay $WEIGHT_DECAY + dropout $DROPOUT"
echo "Total:          $((RUNS * 2)) trainings"
echo ""

echo "============================================================"
echo "[1/2] PLAIN — no regularisation"
echo "============================================================"
echo "+ python scripts/train_lidc.py $COMMON --backbone-dropout 0.0 --tag overfit_plain${SUFFIX}"
$MAMBA_CMD python scripts/train_lidc.py $COMMON \
    --backbone-dropout 0.0 \
    --tag "overfit_plain${SUFFIX}"

echo ""
echo "============================================================"
echo "[2/2] REGULARISED — augment + weight decay + dropout"
echo "============================================================"
echo "+ python scripts/train_lidc.py $COMMON --augment --weight-decay $WEIGHT_DECAY --backbone-dropout $DROPOUT --tag overfit_reg${SUFFIX}"
$MAMBA_CMD python scripts/train_lidc.py $COMMON \
    --augment \
    --weight-decay "$WEIGHT_DECAY" \
    --backbone-dropout "$DROPOUT" \
    --tag "overfit_reg${SUFFIX}"

echo ""
echo "============================================================"
echo "Comparison"
echo "============================================================"
$MAMBA_CMD python scripts/compare_overfit.py --suffix "$SUFFIX"

STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"
exit $STATUS
