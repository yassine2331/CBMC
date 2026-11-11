#!/bin/bash
# =============================================================================
#  run_shapes3d.sh — train every concept-bottleneck VAE on Shapes3D
# =============================================================================
#
#  Trains the generative comparison back to back. All variants share the same
#  ConvVAE encoder/decoder, data subset, split, epochs and seed, so the ONLY
#  difference is the bottleneck between z and the decoder:
#
#    1  vae           plain VAE, no concepts. The image-quality anchor.
#    2  cbm           scalar bottleneck: the decoder sees 6 numbers.
#    3  cem           two anchors, continuous interpolation. OUR METHOD
#                     (Case 3.5).
#    4  cem_linear    one embedding per concept, multiplied by the concept.
#    5  categorical   one anchor per native state, softmax (Case 2).
#    6  binned        categorical over --n-bins bins of each range (Case 3).
#                     shape is nominal, so it keeps its 4 states.
#
#  Concepts are the 6 Shapes3D factors: floor_hue, wall_hue, object_hue,
#  scale, shape, orientation.
#
#
#  PREREQUISITE
#  ------------
#  data/raw/3dshapes/3dshapes.h5 (268 MB, gitignored). Either copy it up:
#      rsync -av data/raw/3dshapes/ <user>@<cluster>:<repo>/data/raw/3dshapes/
#  or download it on the LOGIN node (compute nodes may have no internet):
#      bash run_shapes3d.sh --download
#  The env also needs h5py; it is pip-installed into ./env if missing.
#
#
#  HOW TO RUN
#  ----------
#    sbatch run_shapes3d.sh                          # all six, defaults
#    sbatch run_shapes3d.sh --epochs 50 --n-samples 0    # 0 = all 480k images
#    sbatch run_shapes3d.sh --only "cem binned vae"  # a subset (aliases ok:
#                                                    #  baseline case2 case3 case3.5)
#    sbatch run_shapes3d.sh --n-bins 10 --suffix k10 # keep runs apart
#    (a new --concepts set ALSO needs its own --suffix, or resume refuses)
#    bash   run_shapes3d.sh --parallel               # ONE JOB PER VARIANT, so
#                                                    # they run side by side
#    bash   run_shapes3d.sh --help
#
#  Every flag also reads an env var: EPOCHS, N_SAMPLES, N_BINS, BATCH_SIZE, LR,
#  CONCEPTS, CONCEPT_WEIGHT, INTERVENTION_PROB, CYCLE_WEIGHT, LATENT_DIM, ENC_CHANNELS,
#  KL_WEIGHT, EMBEDDING_DIM, HIDDEN_DIM, DEPTH, SEED, SUFFIX, DEVICE, ONLY.
#
#
#  OPTIONS
#  -------
#    --epochs N              epochs per variant                 default 30
#    --n-samples N           random subset of the 480k images   default 200000
#                            (0 = all; ~6 GB RAM, ~2.4x slower than 200k)
#    --n-bins N              bins per concept for `binned`      default 5
#    --concepts "A B ..."    subset of the 6 factors            default all 6
#                            e.g. --concepts "floor_hue wall_hue object_hue scale orientation"
#    --batch-size N                                             default 256
#    --lr LR                                                    default 1e-3
#    --concept-weight W      weight on the concept loss         default 100
#                            The recon loss is a per-image BCE SUM (~4000) and
#                            the concept loss an MSE/CE (~0.1-2), hence 100.
#    --intervention-prob P   batches decoded from TRUE concepts default 0.5
#    --cycle-weight W        decode shuffled concepts, re-encode, read them
#                            back. This is what makes interventions change the
#                            image; at 0 the decoder learns to ignore the
#                            gate.                              default 100
#    --latent-dim N          z size                             default 32
#    --enc-channels "A B C D"   encoder widths (decoder mirrors) default "32 64 128 256"
#    --kl-weight B           beta                               default 1.0
#    --embedding-dim N       anchor vector size (cem/categorical) default 16
#    --hidden-dim N          anchor MLP width                   default 64
#    --depth N               anchor MLP depth                   default 2
#    --seed N                                                   default 42
#    --suffix NAME           saves to shapes3d_<variant>_NAME
#    --no-resume             retrain from scratch even if a checkpoint exists
#    --parallel              submit one sbatch job per variant and exit
#    --download              fetch 3dshapes.h5 and exit (run on the login node)
#
#
#  RESUMING
#  --------
#  The model is saved after EVERY epoch, and by default a rerun continues
#  each variant from its last saved epoch and skips finished ones. So if the
#  job hits the 24 h wall time, just submit the same command again. Raising
#  --epochs on a finished run trains it further.
#
#
#  COST (rough, from a local run: ~0.55 ms per image per epoch for cem)
#  ----
#  defaults (200k x 30 epochs): vae/cbm ~0.5-1 h, cem/cem_linear ~1 h,
#  binned ~2 h, categorical ~3 h (57 anchor MLPs). Sequential total ~8 h;
#  with --parallel the wall time is the slowest variant.
#
#
#  OUTPUT, and getting it back to the laptop
#  -----------------------------------------
#    outputs/models/shapes3d_<variant>[_suffix]/
#        model.pt  *_config.json  metrics.csv  samples/epoch_XXX.png
#    outputs/logs/shapes3d_<variant>[_suffix].csv   (tracked by git)
#
#  The weights are gitignored (a few MB each). Pull them with:
#      rsync -av <user>@<cluster>:<repo>/outputs/models/shapes3d_* outputs/models/
#  then open notebooks/shapes3d_generation.ipynb, set TRAIN = False and the
#  MODEL / TAG you want; section 7 compares every model in outputs/models/.
#
# =============================================================================

#SBATCH --job-name=cbmc-shapes3d
#SBATCH --time=24:00:00
#SBATCH --partition=blas
#SBATCH --gres=gpu:1g:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --container-image=mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail          # NOT -e: a failing variant must not abort the rest

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

MAMBA_CMD="${MAMBA_CMD:-mamba run -p ./env}"   # override to test locally
H5="data/raw/3dshapes/3dshapes.h5"
H5_URL="https://storage.googleapis.com/3d-shapes/3dshapes.h5"

usage () { sed -n '2,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; }

EPOCHS="${EPOCHS:-60}"
N_SAMPLES="${N_SAMPLES:-300000}"
N_BINS="${N_BINS:-10}"
CONCEPTS="${CONCEPTS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
CONCEPT_WEIGHT="${CONCEPT_WEIGHT:-}"
INTERVENTION_PROB="${INTERVENTION_PROB:-}"
CYCLE_WEIGHT="${CYCLE_WEIGHT:-}"
LATENT_DIM="${LATENT_DIM:-}"
ENC_CHANNELS="${ENC_CHANNELS:-}"
KL_WEIGHT="${KL_WEIGHT:-}"
EMBEDDING_DIM="${EMBEDDING_DIM:-}"
HIDDEN_DIM="${HIDDEN_DIM:-}"
DEPTH="${DEPTH:-}"
SEED="${SEED:-}"
SUFFIX="${SUFFIX:-}"
DEVICE="${DEVICE:-}"
ONLY="${ONLY:-}"
RESUME=1
PARALLEL=""
DOWNLOAD=""

FORWARD=()                # every flag except --only/--parallel, for --parallel
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)             usage; exit 0 ;;
        --only)                ONLY="$2"; shift 2; continue ;;
        --parallel)            PARALLEL=1; shift; continue ;;
        --download)            DOWNLOAD=1; shift; continue ;;
        --no-resume)           RESUME=""; FORWARD+=("$1"); shift; continue ;;
        --epochs)              EPOCHS="$2" ;;
        --n-samples)           N_SAMPLES="$2" ;;
        --n-bins)              N_BINS="$2" ;;
        --concepts)            CONCEPTS="$2" ;;
        --batch-size)          BATCH_SIZE="$2" ;;
        --lr)                  LR="$2" ;;
        --concept-weight)      CONCEPT_WEIGHT="$2" ;;
        --intervention-prob)   INTERVENTION_PROB="$2" ;;
        --cycle-weight)        CYCLE_WEIGHT="$2" ;;
        --latent-dim)          LATENT_DIM="$2" ;;
        --enc-channels)        ENC_CHANNELS="$2" ;;
        --kl-weight)           KL_WEIGHT="$2" ;;
        --embedding-dim)       EMBEDDING_DIM="$2" ;;
        --hidden-dim)          HIDDEN_DIM="$2" ;;
        --depth)               DEPTH="$2" ;;
        --seed)                SEED="$2" ;;
        --suffix)              SUFFIX="$2" ;;
        --device)              DEVICE="$2" ;;
        *) echo "ERROR: unknown option '$1'. Try --help." >&2; exit 1 ;;
    esac
    FORWARD+=("$1" "$2"); shift 2
done

# --- download mode (login node) ---------------------------------------------
if [ -n "$DOWNLOAD" ]; then
    mkdir -p "$(dirname "$H5")"
    if [ -f "$H5" ]; then echo "$H5 already exists."; exit 0; fi
    echo "Downloading $H5_URL -> $H5"
    curl -L --fail -o "$H5.part" "$H5_URL" && mv "$H5.part" "$H5" && ls -lh "$H5"
    exit $?
fi

# --- variants ---------------------------------------------------------------
canon () {                # alias -> variant name, "" if unknown
    case "$1" in
        vae|baseline)                 echo vae ;;
        cbm)                          echo cbm ;;
        cem|case3.5|case3_5)          echo cem ;;
        cem_linear)                   echo cem_linear ;;
        categorical|case2)            echo categorical ;;
        binned|case3)                 echo binned ;;
        *)                            echo "" ;;
    esac
}
ALL="vae cbm cem cem_linear categorical binned"
VARIANTS=""
for v in ${ONLY:-$ALL}; do
    c="$(canon "$v")"
    if [ -z "$c" ]; then
        echo "ERROR: unknown variant '$v'. Choose from: $ALL (or baseline case2 case3 case3.5)" >&2
        exit 1
    fi
    VARIANTS="$VARIANTS $c"
done
VARIANTS="${VARIANTS# }"

# --- parallel mode: one job per variant ---------------------------------------
if [ -n "$PARALLEL" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "ERROR: --parallel submits jobs itself; run it on the login node with" >&2
        echo "       bash run_shapes3d.sh ... --parallel   (not sbatch)" >&2
        exit 1
    fi
    mkdir -p logs
    for v in $VARIANTS; do
        echo "+ sbatch --job-name=cbmc-s3d-$v $0 ${FORWARD[*]} --only $v"
        sbatch --job-name="cbmc-s3d-$v" "$0" "${FORWARD[@]}" --only "$v"
    done
    exit 0
fi

# --- prerequisite checks: fail here, not after the queue wait -----------------
if [ ! -f "$H5" ]; then
    echo "ERROR: $H5 not found." >&2
    echo "Copy it from the laptop:  rsync -av data/raw/3dshapes/ <user>@<cluster>:$PROJECT_DIR/data/raw/3dshapes/" >&2
    echo "or download on the login node:  bash run_shapes3d.sh --download" >&2
    exit 1
fi
if ! $MAMBA_CMD python -c "import h5py" >/dev/null 2>&1; then
    echo "h5py missing from ./env — installing it."
    $MAMBA_CMD pip install -q h5py || { echo "ERROR: could not install h5py into ./env." >&2; exit 1; }
fi

# --- build the python arguments ---------------------------------------------
COMMON="--epochs $EPOCHS --n-samples $N_SAMPLES --n-bins $N_BINS"
[ -n "$CONCEPTS" ]          && COMMON="$COMMON --concepts $CONCEPTS"
[ -n "$BATCH_SIZE" ]        && COMMON="$COMMON --batch-size $BATCH_SIZE"
[ -n "$LR" ]                && COMMON="$COMMON --lr $LR"
[ -n "$CONCEPT_WEIGHT" ]    && COMMON="$COMMON --concept-weight $CONCEPT_WEIGHT"
[ -n "$INTERVENTION_PROB" ] && COMMON="$COMMON --intervention-prob $INTERVENTION_PROB"
[ -n "$CYCLE_WEIGHT" ]      && COMMON="$COMMON --cycle-weight $CYCLE_WEIGHT"
[ -n "$LATENT_DIM" ]        && COMMON="$COMMON --latent-dim $LATENT_DIM"
[ -n "$ENC_CHANNELS" ]      && COMMON="$COMMON --enc-channels $ENC_CHANNELS"
[ -n "$KL_WEIGHT" ]         && COMMON="$COMMON --kl-weight $KL_WEIGHT"
[ -n "$EMBEDDING_DIM" ]     && COMMON="$COMMON --embedding-dim $EMBEDDING_DIM"
[ -n "$HIDDEN_DIM" ]        && COMMON="$COMMON --hidden-dim $HIDDEN_DIM"
[ -n "$DEPTH" ]             && COMMON="$COMMON --depth $DEPTH"
[ -n "$SEED" ]              && COMMON="$COMMON --seed $SEED"
[ -n "$DEVICE" ]            && COMMON="$COMMON --device $DEVICE"
[ -n "$SUFFIX" ]            && COMMON="$COMMON --tag $SUFFIX"
[ -n "$RESUME" ]            && COMMON="$COMMON --resume"

# tqdm redraws fill the .err log; refresh at most every 60 s
export TQDM_MININTERVAL=60
export PYTHONUNBUFFERED=1

N=$(echo $VARIANTS | wc -w | tr -d ' ')
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Host:      $(hostname)"
echo "Start:     $(date)"
echo "Variants:  $N  ($VARIANTS)"
echo "Epochs:    $EPOCHS   n_samples: $N_SAMPLES   n_bins: $N_BINS   resume: ${RESUME:-no}"
echo "========================================"

i=0
RESULTS=""
JOB_T0=$(date +%s)
for v in $VARIANTS; do
    i=$((i + 1))
    T0=$(date +%s)
    echo ""
    echo "============================================================"
    echo "[$i/$N] $v        started $(date +%H:%M:%S)"
    echo "============================================================"
    echo "+ python scripts/train_shapes3d_gen.py --model $v $COMMON"
    $MAMBA_CMD python scripts/train_shapes3d_gen.py --model "$v" $COMMON
    rc=$?
    MINS=$(( ($(date +%s) - T0) / 60 ))
    if [ $rc -eq 0 ]; then
        RESULTS="$RESULTS\n  $v: OK (${MINS} min)"
    else
        echo "[$i/$N] $v FAILED (exit $rc) — continuing" >&2
        RESULTS="$RESULTS\n  $v: FAILED exit $rc (${MINS} min)"
    fi
done

echo ""
echo "========================================"
echo "Finished after $(( ($(date +%s) - JOB_T0) / 60 )) min"
printf "$RESULTS\n"
echo ""
echo "Models:  $PROJECT_DIR/outputs/models/shapes3d_*"
echo "Fetch:   rsync -av $USER@<cluster>:$PROJECT_DIR/outputs/models/shapes3d_* outputs/models/"
echo "End:     $(date)"
echo "========================================"
case "$RESULTS" in *FAILED*) exit 1 ;; esac
exit 0
