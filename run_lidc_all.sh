#!/bin/bash
# =============================================================================
#  run_lidc_all.sh — the full concept-representation comparison on LIDC
# =============================================================================
#
#  Runs SEVEN variants back to back and builds one comparison table. They all
#  share the same encoder, the same split, the same epochs and the same seeds,
#  so the ONLY difference is how the concept is represented.
#
#    1  nn                 plain 3D CNN, no bottleneck. The accuracy anchor:
#                          what you can reach with no interpretability at all.
#    2  cbm                scalar bottleneck. One number per concept, no anchor
#                          embeddings, so the head sees only k numbers.
#    3  cem_binary         CEM with concepts collapsed to high/low, split at the
#                          TRAIN mean. The classic binary Case 1.
#    4  categorical        one anchor per state, softmax over them. Ratings keep
#                          their 5 levels; diameter/volume/surface_area get
#                          --n-bins bins. (Case 2 / Case 3.)
#    5  cem_linear_norm    one embedding per concept, multiplied by the concept.
#                          Concepts normalised to [-1,1].
#    6  cem_linear_raw     same, but concepts passed in RAW units. Volume runs
#                          to ~1e4, so the gate is enormous — this is the
#                          control showing why normalisation matters.
#    7  cem                two anchors, continuous interpolation. OUR METHOD
#                          (Case 3.5).
#
#
#  HOW TO RUN
#  ----------
#    sbatch run_lidc_all.sh                       # all seven, defaults
#    sbatch run_lidc_all.sh --epochs 100 --runs 5
#    sbatch run_lidc_all.sh --only "cem cbm nn"   # a subset
#    sbatch run_lidc_all.sh --n-bins 10 --augment
#    bash   run_lidc_all.sh --help
#
#  Env vars work too: EPOCHS, RUNS, N_BINS, CONCEPT_WEIGHT, INTERVENTION_PROB,
#  CONV_CHANNELS, EMBEDDING_DIM, AUGMENT, SUFFIX, DEVICE, ONLY.
#
#
#  OPTIONS
#  -------
#    --epochs N            epochs per variant                 default 60
#    --runs N              seeds per variant (mean +/- std)   default 5
#    --n-bins N            bins for continuous concepts in
#                          the categorical variant            default 5
#    --concept-weight W    weight on the concept loss         default from config
#                          (task is CE ~0.7, concept MSE ~0.15, so 1.0 is sane)
#    --intervention-prob P batches trained with true concepts default from config
#    --conv-channels "A B C D"  encoder width                 default "16 32 64 128"
#    --embedding-dim N     anchor vector size                 default 16
#    --augment             random flips / rotations (recommended, ~940 samples)
#    --only "a b c"        run just these variant keys
#    --suffix S            appended to every output filename, to keep sweeps apart
#    --device D            cuda | mps | cpu
#
#
#  OUTPUT
#  ------
#    outputs/results/lidc_<variant><suffix>.csv   one per variant, rewritten
#                                                 after EVERY seed finishes
#    outputs/results/lidc_summary<suffix>.csv     merged table, refreshed after
#                                                 every variant finishes
#    outputs/logs/lidc_<variant><suffix>_history.csv
#                                                 per-epoch training curve:
#                                                 variant, seed, epoch,
#                                                 task_loss, concept_loss,
#                                                 train_acc, seconds
#
#  Nothing is written only at the end, so killing the job still leaves every
#  finished seed and every epoch recorded. The history file is what to plot to
#  compare how hard each representation is to train.
#
#  The table reports accuracy, balanced accuracy (USE THIS — classes are 56/44),
#  intervention accuracy, concept accuracy where meaningful, concept MAE
#  relative to each concept's range, and parameter count.
#
#  Concept error is always measured in ORIGINAL units against the TRUE raw
#  values, even for binary and categorical. That is deliberate: it charges each
#  representation for the information it threw away.
#
#
#  COST
#  ----
#  ~35 s/epoch on a laptop GPU, faster on the cluster. Seven variants x 60
#  epochs x 5 seeds is roughly 10-20 hours — close to the 24 h wall time, so
#  use --runs 3 or split with --only if the queue is tight.
#
#
#  SANITY CHECKS
#  -------------
#    - balanced accuracy clearly above 0.500 (else the model collapsed)
#    - the plain CNN (variant 1) should be the accuracy ceiling; if a
#      bottleneck beats it, the concepts are a useful inductive bias
#    - intervention accuracy >= plain accuracy; if not, raise
#      --intervention-prob
#    - cem_linear_raw is EXPECTED to do badly. That is the point.
#
# =============================================================================

#SBATCH --job-name=cbmc-lidc-all
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

EPOCHS="${EPOCHS:-60}"
RUNS="${RUNS:-5}"
N_BINS="${N_BINS:-5}"
CONCEPT_WEIGHT="${CONCEPT_WEIGHT:-}"
INTERVENTION_PROB="${INTERVENTION_PROB:-}"
CONV_CHANNELS="${CONV_CHANNELS:-}"
EMBEDDING_DIM="${EMBEDDING_DIM:-}"
AUGMENT="${AUGMENT:-}"
ONLY="${ONLY:-}"
SUFFIX="${SUFFIX:-}"
DEVICE="${DEVICE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)            usage; exit 0 ;;
        --epochs)             EPOCHS="$2"; shift 2 ;;
        --runs)               RUNS="$2"; shift 2 ;;
        --n-bins)             N_BINS="$2"; shift 2 ;;
        --concept-weight)     CONCEPT_WEIGHT="$2"; shift 2 ;;
        --intervention-prob)  INTERVENTION_PROB="$2"; shift 2 ;;
        --conv-channels)      CONV_CHANNELS="$2"; shift 2 ;;
        --embedding-dim)      EMBEDDING_DIM="$2"; shift 2 ;;
        --augment)            AUGMENT=1; shift ;;
        --only)               ONLY="$2"; shift 2 ;;
        --suffix)             SUFFIX="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
done

if [ ! -f data/processed/lidc/nodules.csv ]; then
    echo "ERROR: data/processed/lidc/nodules.csv not found." >&2
    echo "Build the cubes first:  python scripts/prepare_lidc.py --all" >&2
    exit 1
fi

# variant key -> flags for scripts/train_lidc.py
variant_args () {
    case "$1" in
        nn)               echo "--bottleneck none" ;;
        cbm)              echo "--bottleneck cbm" ;;
        cem_binary)       echo "--bottleneck cem --concept-mode binary" ;;
        categorical)      echo "--bottleneck categorical --concept-mode categorical --n-bins $N_BINS" ;;
        cem_linear_norm)  echo "--bottleneck cem_linear --concept-scaling minmax" ;;
        cem_linear_raw)   echo "--bottleneck cem_linear --concept-scaling raw" ;;
        cem)              echo "--bottleneck cem" ;;
        *) echo "" ;;
    esac
}

ALL="nn cbm cem_binary categorical cem_linear_norm cem_linear_raw cem"
VARIANTS="${ONLY:-$ALL}"
for v in $VARIANTS; do
    if [ -z "$(variant_args "$v")" ]; then
        echo "ERROR: unknown variant '$v'." >&2
        echo "Choose from: $ALL" >&2
        exit 1
    fi
done

COMMON="--epochs $EPOCHS --runs $RUNS"
[ -n "$CONCEPT_WEIGHT" ]    && COMMON="$COMMON --concept-weight $CONCEPT_WEIGHT"
[ -n "$INTERVENTION_PROB" ] && COMMON="$COMMON --intervention-prob $INTERVENTION_PROB"
[ -n "$CONV_CHANNELS" ]     && COMMON="$COMMON --conv-channels $CONV_CHANNELS"
[ -n "$EMBEDDING_DIM" ]     && COMMON="$COMMON --embedding-dim $EMBEDDING_DIM"
[ -n "$AUGMENT" ]           && COMMON="$COMMON --augment"
[ -n "$DEVICE" ]            && COMMON="$COMMON --device $DEVICE"

N=$(echo $VARIANTS | wc -w | tr -d ' ')
echo "Variants:  $N  ($VARIANTS)"
echo "Epochs:    $EPOCHS   Runs: $RUNS   n_bins: $N_BINS"
echo "Total:     $((N * RUNS)) trainings"
echo ""

i=0
for v in $VARIANTS; do
    i=$((i + 1))
    echo ""
    echo "============================================================"
    echo "[$i/$N] $v"
    echo "============================================================"
    echo "+ python scripts/train_lidc.py $(variant_args "$v") $COMMON --tag ${v}${SUFFIX}"
    $MAMBA_CMD python scripts/train_lidc.py $(variant_args "$v") $COMMON \
        --tag "${v}${SUFFIX}"

    # Refresh the merged table after every variant, so a job that dies or runs
    # out of wall time still leaves a usable summary of what finished.
    $MAMBA_CMD python scripts/aggregate_lidc.py --suffix "$SUFFIX" >/dev/null 2>&1 || true
    echo "  -> summary refreshed: outputs/results/lidc_summary${SUFFIX}.csv"
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
$MAMBA_CMD python scripts/aggregate_lidc.py --suffix "$SUFFIX"

STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"
exit $STATUS
