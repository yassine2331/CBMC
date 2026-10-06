#!/bin/bash
# =============================================================================
#  run_lidc.sh — train a concept bottleneck on LIDC nodule cubes
# =============================================================================
#
#  WHAT IT TRAINS
#  --------------
#      cube (1,64,64,64)  ->  Conv3D encoder  ->  concepts  ->  malignant/benign
#
#  Concepts are 9 continuous nodule properties: 6 radiologist ratings
#  (subtlety, sphericity, margin, lobulation, spiculation, texture) and 3
#  geometric measurements (diameter mm, volume mm3, surface_area mm2).
#  All are min-max scaled to [-1,1] using TRAIN statistics only, and the
#  scaler is inverted at test time so concept error is reported in real units.
#
#  calcification and internalStructure are deliberately excluded: they are
#  nominal codes, not magnitudes. Use --bottleneck categorical for those.
#
#  Splits are BY PATIENT. 71% of nodules share a patient with another and most
#  siblings share a label, so a nodule-level split leaks the scanner instead of
#  testing the model. The script asserts there is no overlap.
#
#
#  PREREQUISITE
#  ------------
#  The cubes must exist:  data/processed/lidc/{cubes/*.npy, nodules.csv}
#  Build them with:       python scripts/prepare_lidc.py --all --verbose
#  (downloads one scan at a time and deletes it; ~1.4 GB kept, not 128 GB)
#
#
#  HOW TO RUN
#  ----------
#    sbatch run_lidc.sh                                 # defaults below
#    sbatch run_lidc.sh --epochs 100 --runs 5
#    sbatch run_lidc.sh --bottleneck none --tag baseline      # no-bottleneck anchor
#    sbatch run_lidc.sh --bottleneck cem_linear --tag ablation
#    sbatch run_lidc.sh --conv-channels "32 64 128 256"       # bigger encoder
#    sbatch run_lidc.sh --embedding-dim 32 --hidden-dim 128   # bigger bottleneck
#    bash   run_lidc.sh --help
#
#  Every flag also reads an env var: EPOCHS, RUNS, BOTTLENECK, CONV_CHANNELS,
#  EMBEDDING_DIM, HIDDEN_DIM, DEPTH, LR, BATCH_SIZE, CONCEPT_WEIGHT,
#  INTERVENTION_PROB, MIN_ANNOTATIONS, AUGMENT, TAG, DEVICE, SEED.
#
#
#  WHERE THE SETTINGS LIVE
#  -----------------------
#  Defaults are in experiments/configs/ and flags override them, so a sweep
#  needs no file edits:
#
#    lidc_backbone.json   conv_channels, kernel_size, batch_norm, dropout
#    lidc_cem.json        n_concepts, embedding_dim, hidden_dim, depth, dropout
#    lidc_train.json      epochs, lr, batch_size, concept_weight,
#                         intervention_prob
#    lidc_data.json       which nodules (min_annotations), which concepts,
#                         HU window, test_size
#
#  Edit the JSON to change a default permanently; pass a flag for one run.
#
#
#  THE SETTINGS THAT MATTER
#  ------------------------
#    --bottleneck NAME       cem            Case 3.5, two anchors (default)
#                            categorical    Case 2, n anchors + softmax
#                            cem_tanh       single embedding, tanh gate
#                            cem_linear     single embedding, linear gate
#                            cem_linear_raw as above, no output LayerNorm
#                            none           no bottleneck — accuracy anchor
#
#    --conv-channels "A B C D"   encoder width per block. Each block halves the
#                            cube, so 4 blocks take 64 -> 4. More channels =
#                            more capacity and much more memory.
#                            default "16 32 64 128"  (~591K params, 2.4 MB)
#
#    --embedding-dim N       size of each concept anchor vector.   default 16
#                            The head input is embedding_dim x n_concepts.
#
#    --concept-weight W      weight on the concept loss.           default 1.0
#                            Here the task loss is a cross-entropy (~0.7) and
#                            the concept loss an MSE in [-1,1] (~0.15), so they
#                            are within ~4x and 1.0 is sane. Watch the printed
#                            losses: if concept is far smaller, raise this.
#                            (On ArithmeticMNIST the task MSE spanned 0-81 and
#                            swamped concepts 95:1, which needed W=100.)
#
#    --intervention-prob P   fraction of batches trained with TRUE concepts.
#                                                                  default 0.25
#                            At 0 a test-time intervention is out-of-distribution
#                            and HURTS. 0.25 was enough elsewhere to make
#                            intervention accuracy near-perfect.
#
#    --min-annotations N     drop nodules seen by fewer radiologists. default 3
#                            3 -> ~1180 nodules, 56/44 class balance
#                            1 -> ~1990 nodules, 68/32 balance, noisier labels
#
#    --augment               random flips and 90-degree rotations. Nodules have
#                            no canonical orientation so this is free and the
#                            dataset is small (~940 training samples).
#
#    --no-class-weight       disable class balancing (ON by default — without
#                            it the model collapses to predicting all benign).
#
#    --epochs / --lr / --batch-size / --hidden-dim / --depth / --runs / --seed
#    --tag NAME              suffix for the output CSV, to keep runs apart.
#
#
#  WHAT YOU GET BACK
#  -----------------
#  Printed table and a CSV at outputs/results/lidc[_<tag>].csv, mean +/- std
#  over seeds:
#
#    accuracy               fraction of nodules classified correctly
#    balanced_accuracy      mean of per-class accuracy; use THIS, the classes
#                           are 56/44 and plain accuracy flatters a collapse
#    intervention_accuracy  accuracy when the TRUE concepts are fed in. The
#                           interpretability number: does correcting a concept
#                           fix the prediction?
#    concept_mae / _mse     concept error in ORIGINAL units (mm, mm3, rating
#                           points), plus a per-concept breakdown
#    params / size_mb       model size
#
#
#  SANITY CHECKS BEFORE TRUSTING A RUN
#  -----------------------------------
#    - balanced accuracy clearly above 0.500, and the confusion matrix is not
#      all one class
#    - concept MAE below the mean-predictor baseline (diameter ~5.7 mm on this
#      split); if not, the bottleneck is carrying nothing
#    - intervention accuracy >= plain accuracy; if lower, raise
#      --intervention-prob
#    - the "patient leak!" assertion did not fire
#
#
#  COST
#  ----
#  ~35 s/epoch on an M-series laptop GPU at batch 16 with ~940 training
#  samples; faster on a cluster GPU. 60 epochs x 5 seeds is a few hours.
#
#
#  RELATED FILES
#  -------------
#    scripts/train_lidc.py          the trainer this submits
#    scripts/prepare_lidc.py        builds the cubes from TCIA
#    architectures/conv3d_cem.py    the 3D encoder + bottleneck wiring
#    notebooks/lidc_cubes.ipynb     data inspection and the same pipeline inline
#    run_binned.sh                  the ArithmeticMNIST binning experiment
#
# =============================================================================

#SBATCH --job-name=cbmc-lidc
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
echo "Directory: $PROJECT_DIR"
echo "========================================"

MAMBA_CMD="mamba run -p ./env"

usage () { sed -n '2,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; }

BOTTLENECK="${BOTTLENECK:-cem}"
RUNS="${RUNS:-5}"
SEED="${SEED:-41}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
CONCEPT_WEIGHT="${CONCEPT_WEIGHT:-}"
INTERVENTION_PROB="${INTERVENTION_PROB:-}"
CONV_CHANNELS="${CONV_CHANNELS:-}"
EMBEDDING_DIM="${EMBEDDING_DIM:-}"
HIDDEN_DIM="${HIDDEN_DIM:-}"
DEPTH="${DEPTH:-}"
MIN_ANNOTATIONS="${MIN_ANNOTATIONS:-}"
AUGMENT="${AUGMENT:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
BACKBONE_DROPOUT="${BACKBONE_DROPOUT:-}"
NO_CLASS_WEIGHT="${NO_CLASS_WEIGHT:-}"
TAG="${TAG:-}"
DEVICE="${DEVICE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)             usage; exit 0 ;;
        --bottleneck)          BOTTLENECK="$2"; shift 2 ;;
        --runs)                RUNS="$2"; shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --epochs)              EPOCHS="$2"; shift 2 ;;
        --lr)                  LR="$2"; shift 2 ;;
        --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
        --concept-weight)      CONCEPT_WEIGHT="$2"; shift 2 ;;
        --intervention-prob)   INTERVENTION_PROB="$2"; shift 2 ;;
        --conv-channels)       CONV_CHANNELS="$2"; shift 2 ;;
        --embedding-dim)       EMBEDDING_DIM="$2"; shift 2 ;;
        --hidden-dim)          HIDDEN_DIM="$2"; shift 2 ;;
        --depth)               DEPTH="$2"; shift 2 ;;
        --min-annotations)     MIN_ANNOTATIONS="$2"; shift 2 ;;
        --augment)             AUGMENT=1; shift ;;
        --weight-decay)        WEIGHT_DECAY="$2"; shift 2 ;;
        --backbone-dropout)    BACKBONE_DROPOUT="$2"; shift 2 ;;
        --no-class-weight)     NO_CLASS_WEIGHT=1; shift ;;
        --tag)                 TAG="$2"; shift 2 ;;
        --device)              DEVICE="$2"; shift 2 ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
done

# --- prerequisite check: fail here, not after the queue wait ---
if [ ! -f data/processed/lidc/nodules.csv ]; then
    echo "ERROR: data/processed/lidc/nodules.csv not found." >&2
    echo "Build the cubes first:  python scripts/prepare_lidc.py --all" >&2
    exit 1
fi
N_CUBES=$(ls data/processed/lidc/cubes/*.npy 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_CUBES" -eq 0 ]; then
    echo "ERROR: no cubes in data/processed/lidc/cubes/." >&2; exit 1
fi

case "$BOTTLENECK" in
    cem|cem_tanh|cem_linear|cem_linear_raw|categorical|none) ;;
    *) echo "ERROR: unknown --bottleneck '$BOTTLENECK'." >&2
       echo "Choose: cem, cem_tanh, cem_linear, cem_linear_raw, categorical, none" >&2
       exit 1 ;;
esac

for pair in "RUNS:$RUNS" "SEED:$SEED"; do
    n="${pair%%:*}"; v="${pair#*:}"
    if ! [[ "$v" =~ ^[0-9]+$ ]] || [ "$v" -lt 1 ]; then
        echo "ERROR: $n must be an integer >= 1 (got '$v')." >&2; exit 1
    fi
done

ARGS="--bottleneck $BOTTLENECK --runs $RUNS --seed $SEED"
[ -n "$EPOCHS" ]            && ARGS="$ARGS --epochs $EPOCHS"
[ -n "$LR" ]                && ARGS="$ARGS --lr $LR"
[ -n "$BATCH_SIZE" ]        && ARGS="$ARGS --batch-size $BATCH_SIZE"
[ -n "$CONCEPT_WEIGHT" ]    && ARGS="$ARGS --concept-weight $CONCEPT_WEIGHT"
[ -n "$INTERVENTION_PROB" ] && ARGS="$ARGS --intervention-prob $INTERVENTION_PROB"
[ -n "$CONV_CHANNELS" ]     && ARGS="$ARGS --conv-channels $CONV_CHANNELS"
[ -n "$EMBEDDING_DIM" ]     && ARGS="$ARGS --embedding-dim $EMBEDDING_DIM"
[ -n "$HIDDEN_DIM" ]        && ARGS="$ARGS --hidden-dim $HIDDEN_DIM"
[ -n "$DEPTH" ]             && ARGS="$ARGS --depth $DEPTH"
[ -n "$MIN_ANNOTATIONS" ]   && ARGS="$ARGS --min-annotations $MIN_ANNOTATIONS"
[ -n "$AUGMENT" ]           && ARGS="$ARGS --augment"
[ -n "$WEIGHT_DECAY" ]      && ARGS="$ARGS --weight-decay $WEIGHT_DECAY"
[ -n "$BACKBONE_DROPOUT" ]  && ARGS="$ARGS --backbone-dropout $BACKBONE_DROPOUT"
[ -n "$NO_CLASS_WEIGHT" ]   && ARGS="$ARGS --no-class-weight"
[ -n "$TAG" ]               && ARGS="$ARGS --tag $TAG"
[ -n "$DEVICE" ]            && ARGS="$ARGS --device $DEVICE"

echo "Cubes available:   $N_CUBES"
echo "Bottleneck:        $BOTTLENECK"
echo "Runs (seeds):      $RUNS  (starting at $SEED)"
echo ""
echo "+ python scripts/train_lidc.py $ARGS"
echo ""

$MAMBA_CMD python scripts/train_lidc.py $ARGS

STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"
exit $STATUS
