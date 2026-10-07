#!/bin/bash
# =============================================================================
#  run_lidc_incomplete.sh — concept-incompleteness test on LIDC
# =============================================================================
#
#  On the full 9-concept set, the concepts alone already predict malignancy as
#  well as the image does (logistic regression on TRUE concepts: balanced acc
#  0.842, plain CNN: 0.841). A CBM then loses nothing, so it ties with CEM.
#
#  This script trains the same models on a SMALLER concept set that you pick.
#  The image still shows everything, but the bottleneck is told about fewer
#  concepts. Expected if the theory holds:
#    cbm   drops toward the concept-only ceiling of the chosen subset
#    cem   stays near the plain CNN, because the embeddings carry the rest
#
#  Concept-only ceilings (logistic regression on true concepts, 15 grouped
#  splits), to compare CBM against:
#    all 9 default concepts                              0.842
#    no size (6 ratings)                                 0.799
#    subtlety sphericity texture                         0.739
#
#
#  AVAILABLE CONCEPTS (pick any subset with --concepts "a b c")
#  ------------------
#    subtlety            radiologist rating 1-5   how hard the nodule is to see
#    sphericity          radiologist rating 1-5   linear ... round
#    margin              radiologist rating 1-5   poorly ... sharply defined
#    lobulation          radiologist rating 1-5   none ... marked
#    spiculation         radiologist rating 1-5   none ... marked
#    texture             radiologist rating 1-5   non-solid ... solid
#    diameter            measured, mm             size  (alone predicts 0.831!)
#    volume              measured, mm^3           size
#    surface_area        measured, mm^2           size
#    internalStructure   radiologist rating 1-4   NOT in the default set
#    calcification       radiologist rating 1-6   NOT in the default set
#
#  malignancy is NOT available: the label is derived from it.
#
#  Suggested sets:
#    --concepts "subtlety sphericity margin lobulation spiculation texture"
#                                                    drop the size concepts
#    --concepts "subtlety sphericity texture"        drop size + shape cues
#
#
#  HOW TO RUN
#  ----------
#    sbatch run_lidc_incomplete.sh --concepts "subtlety sphericity texture"
#    sbatch run_lidc_incomplete.sh --concepts "..." --only "cbm cem nn"
#    bash   run_lidc_incomplete.sh --help
#
#  Defaults match the current final LIDC run: 250 epochs, 15 seeds,
#  --augment, --weight-decay 0.01, --backbone-dropout 0.1. Everything else
#  (concept weight, intervention prob, widths) comes from the configs, same as
#  run_lidc_all.sh, so the numbers are directly comparable to lidc_summary.csv.
#
#
#  OPTIONS
#  -------
#    --concepts "a b c"    REQUIRED. Concept subset, names from the list above.
#    --only "a b c"        variants to run                   default "cbm cem"
#                          Choose from: nn cbm cem_binary categorical
#                          cem_linear_norm cem_linear_raw cem
#                          (nn ignores concepts entirely, so its result in
#                          lidc_nn.csv already applies; add it only as a check)
#    --epochs N            epochs per variant                default 250
#    --runs N              seeds per variant                 default 15
#    --n-bins N            bins for the categorical variant  default 5
#    --concept-weight W    weight on the concept loss        default from config
#    --intervention-prob P                                   default from config
#    --conv-channels "A B C D"                               default from config
#    --embedding-dim N                                       default from config
#    --no-augment          turn augmentation off             (on by default)
#    --weight-decay W                                        default 0.01
#    --backbone-dropout D                                    default 0.1
#    --min-annotations N                                     default from config
#    --suffix S            output suffix. Default is built from the concept
#                          names, e.g. _inc_subtlety-sphericity-texture
#    --device D            cuda | mps | cpu
#
#
#  OUTPUT
#  ------
#    outputs/results/lidc_<variant><suffix>.csv     one per variant, rewritten
#                                                   after every seed; it has a
#                                                   "concepts" column
#    outputs/results/lidc_summary<suffix>.csv       merged table
#    outputs/logs/lidc_<variant><suffix>_history.csv
#
# =============================================================================

#SBATCH --job-name=cbmc-lidc-incomplete
#SBATCH --time=24:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --container-image=mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

MAMBA_CMD="mamba run -p ./env"
usage () { sed -n '4,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; }

CONCEPTS="${CONCEPTS:-}"
ONLY="${ONLY:-cbm cem}"
EPOCHS="${EPOCHS:-250}"
RUNS="${RUNS:-15}"
N_BINS="${N_BINS:-5}"
CONCEPT_WEIGHT="${CONCEPT_WEIGHT:-}"
INTERVENTION_PROB="${INTERVENTION_PROB:-}"
CONV_CHANNELS="${CONV_CHANNELS:-}"
EMBEDDING_DIM="${EMBEDDING_DIM:-}"
AUGMENT="${AUGMENT:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BACKBONE_DROPOUT="${BACKBONE_DROPOUT:-0.1}"
MIN_ANNOTATIONS="${MIN_ANNOTATIONS:-}"
SUFFIX="${SUFFIX:-}"
DEVICE="${DEVICE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)            usage; exit 0 ;;
        --concepts)           CONCEPTS="$2"; shift 2 ;;
        --only)               ONLY="$2"; shift 2 ;;
        --epochs)             EPOCHS="$2"; shift 2 ;;
        --runs)               RUNS="$2"; shift 2 ;;
        --n-bins)             N_BINS="$2"; shift 2 ;;
        --concept-weight)     CONCEPT_WEIGHT="$2"; shift 2 ;;
        --intervention-prob)  INTERVENTION_PROB="$2"; shift 2 ;;
        --conv-channels)      CONV_CHANNELS="$2"; shift 2 ;;
        --embedding-dim)      EMBEDDING_DIM="$2"; shift 2 ;;
        --augment)            AUGMENT=1; shift ;;
        --no-augment)         AUGMENT=""; shift ;;
        --weight-decay)       WEIGHT_DECAY="$2"; shift 2 ;;
        --backbone-dropout)   BACKBONE_DROPOUT="$2"; shift 2 ;;
        --min-annotations)    MIN_ANNOTATIONS="$2"; shift 2 ;;
        --suffix)             SUFFIX="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
done

if [ -z "$CONCEPTS" ]; then
    echo "ERROR: --concepts is required, e.g. --concepts \"subtlety sphericity texture\"." >&2
    echo "See --help for the list of available concepts." >&2
    exit 1
fi
if [ ! -f data/processed/lidc/nodules.csv ]; then
    echo "ERROR: data/processed/lidc/nodules.csv not found." >&2
    echo "Build the cubes first:  python scripts/prepare_lidc.py --all" >&2
    exit 1
fi

# Default suffix names the concept set, so different sets never overwrite
# each other or the main results.
[ -z "$SUFFIX" ] && SUFFIX="_inc_$(echo $CONCEPTS | tr ' ' '-')"

# variant key -> flags for scripts/train_lidc.py (same as run_lidc_all.sh)
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
for v in $ONLY; do
    if [ -z "$(variant_args "$v")" ]; then
        echo "ERROR: unknown variant '$v'. Choose from: $ALL" >&2
        exit 1
    fi
done

COMMON="--epochs $EPOCHS --runs $RUNS --concepts $CONCEPTS"
[ -n "$CONCEPT_WEIGHT" ]    && COMMON="$COMMON --concept-weight $CONCEPT_WEIGHT"
[ -n "$INTERVENTION_PROB" ] && COMMON="$COMMON --intervention-prob $INTERVENTION_PROB"
[ -n "$CONV_CHANNELS" ]     && COMMON="$COMMON --conv-channels $CONV_CHANNELS"
[ -n "$EMBEDDING_DIM" ]     && COMMON="$COMMON --embedding-dim $EMBEDDING_DIM"
[ -n "$AUGMENT" ]           && COMMON="$COMMON --augment"
[ -n "$WEIGHT_DECAY" ]      && COMMON="$COMMON --weight-decay $WEIGHT_DECAY"
[ -n "$BACKBONE_DROPOUT" ]  && COMMON="$COMMON --backbone-dropout $BACKBONE_DROPOUT"
[ -n "$MIN_ANNOTATIONS" ]   && COMMON="$COMMON --min-annotations $MIN_ANNOTATIONS"
[ -n "$DEVICE" ]            && COMMON="$COMMON --device $DEVICE"

N=$(echo $ONLY | wc -w | tr -d ' ')
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "Concepts:  $CONCEPTS"
echo "Variants:  $N  ($ONLY)"
echo "Epochs:    $EPOCHS   Runs: $RUNS   augment: ${AUGMENT:-off}"
echo "           weight decay: $WEIGHT_DECAY   backbone dropout: $BACKBONE_DROPOUT"
echo "Suffix:    $SUFFIX"
echo "========================================"

i=0
for v in $ONLY; do
    i=$((i + 1))
    echo ""
    echo "============================================================"
    echo "[$i/$N] $v   concepts: $CONCEPTS"
    echo "============================================================"
    echo "+ python scripts/train_lidc.py $(variant_args "$v") $COMMON --tag ${v}${SUFFIX}"
    $MAMBA_CMD python scripts/train_lidc.py $(variant_args "$v") $COMMON \
        --tag "${v}${SUFFIX}"

    $MAMBA_CMD python scripts/aggregate_lidc.py --suffix "$SUFFIX" >/dev/null 2>&1 || true
    echo "  -> summary refreshed: outputs/results/lidc_summary${SUFFIX}.csv"
done

echo ""
echo "============================================================"
echo "Summary   (concepts: $CONCEPTS)"
echo "============================================================"
$MAMBA_CMD python scripts/aggregate_lidc.py --suffix "$SUFFIX"

STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"
exit $STATUS
