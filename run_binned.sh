#!/bin/bash
# =============================================================================
#  run_binned.sh — Binned concepts (Case 3) vs continuous interpolation (3.5)
# =============================================================================
#
#  WHAT THIS EXPERIMENT ASKS
#  -------------------------
#  ArithmeticMNIST concepts are the two digit values, integers in [1, 9].
#
#    Case 3.5 (CEM)      treats a concept as CONTINUOUS. Two anchor embeddings
#                        per concept; a predicted scalar slides between them.
#    Case 3   (binned)   chops [1, 9] into k bins and treats "which bin" as a
#                        CATEGORICAL concept. k anchors per concept, softmax
#                        over them.
#
#  Both are trained identically — same backbone, same embedding size, same
#  learning rate — so the only difference is how the concept is represented.
#  The question is what discretising a continuous concept costs, and whether
#  more bins help.
#
#
#  HOW TO RUN IT
#  -------------
#    sbatch run_binned.sh                             # the defaults below
#    sbatch run_binned.sh --epochs 50 --runs 5
#    sbatch run_binned.sh --bins "5 10 20 40"
#    sbatch run_binned.sh --intervention-prob 0.5
#    sbatch run_binned.sh --concept-weight 1 --tag cw1
#    bash   run_binned.sh --help                      # prints usage, runs nothing
#
#  Every flag also reads an environment variable, so these are equivalent:
#    sbatch run_binned.sh --epochs 50
#    EPOCHS=50 sbatch run_binned.sh
#
#
#  THE PARAMETERS
#  --------------
#    --epochs N              training epochs per model.            default 30
#                            20 is roughly where this task starts working; at
#                            5 epochs nothing has converged and all models look
#                            identical. Do not go below 20.
#
#    --runs N                seeds per model.                      default 5
#                            >1 reports mean +/- std. Seeds are 41, 42, 43...
#
#    --bins "A B C"          bin counts to sweep.                  default "5 10 20"
#                            Concepts are digits 1-9, so k=10 is about one bin
#                            per digit, k=5 is coarse (2 digits per bin), k=20
#                            is finer than the data (some bins stay empty).
#
#    --concept-weight W      weight on the concept loss.           default 100
#
#                            *** READ THIS BEFORE CHANGING IT ***
#                            The task target is an arithmetic result spanning
#                            0-81, so its MSE is ~30. The concept target is
#                            normalised to [-1,1], so its MSE is ~0.3. With
#                            W=1 the task gradient is ~95x bigger and the
#                            concept head is effectively UNSUPERVISED: concept
#                            accuracy stays at chance and every model looks
#                            equally bad. W=100 is what made concepts actually
#                            get learned (k=10 went from 13% to 94% concept
#                            accuracy). W was never properly tuned — only
#                            1, 10 and 100 were tried.
#
#    --intervention-prob P   fraction of training batches that use ground-truth
#                            concepts instead of predicted ones.   default 0.25
#                            At P=0 the model only ever sees soft predicted
#                            concepts, so a hard test-time intervention is
#                            out-of-distribution and HURTS. P=0.25 was enough
#                            to make intervention accuracy near-perfect.
#
#    --seed N                first seed.                           default 41
#    --tag NAME              suffix for the output CSV, to keep runs apart.
#    --device D              cuda | mps | cpu.                     default auto
#
#
#  WHAT YOU GET BACK
#  -----------------
#  A table printed to the log, and a CSV at
#      outputs/results/binned_concepts[_<tag>].csv
#
#  Columns, all on the TEST set, all mean +/- std over seeds:
#
#    task mse           regression error on the arithmetic result. Lower better.
#    concept mse        error on the concepts, in normalised [-1,1] space so
#                       Case 3 and Case 3.5 are directly comparable. For binned
#                       models the predicted bin's CENTRE is decoded back to a
#                       value first — that decode is the quantisation cost.
#    intervention mse   task error when both concepts are replaced by the truth.
#                       This is the interpretability metric that matters: it
#                       says whether correcting a concept actually fixes the
#                       prediction.
#    concept acc        fraction of concepts predicted exactly right. NOT
#                       comparable across rows — for binned models it is "right
#                       bin out of k", for Case 3.5 it is "rounds to the right
#                       digit". Compare concept mse instead; use this only to
#                       check a model is above chance (1/k).
#    params / size MB   model size. Binning costs k anchor networks per concept
#                       instead of 2, so this grows linearly with k.
#
#  Also printed: the QUANTISATION FLOOR per bin count — the concept MSE a
#  perfect binned model still cannot beat, because bin centres are not the
#  true values. If a model's concept mse is far above its floor, the bins are
#  not the limitation; training is.
#
#
#  WHAT WAS FOUND LAST TIME  (30 epochs, 1 seed, concept-weight 100)
#  -----------------------------------------------------------------
#    model                task_mse  concept_mse  interv_mse  concept_acc  params
#    CEM (Case 3.5)         29.00       0.0723       16.98        0.426   159K
#    Binned k=5             26.34       0.0925        7.45        0.897   247K
#    Binned k=10            13.15       0.0357        2.58        0.944   392K
#    Binned k=20            10.09       0.0313        2.19        0.948   683K
#
#  Binning won on every metric and improved monotonically with k, but cost 4x
#  the parameters at k=20. Single seed, so treat as indicative, not final —
#  that is what --runs 5 is for.
#
#  The SAME sweep at --concept-weight 1 showed the opposite (all binned models
#  worse than a model that ignores the image). That was an artefact of the
#  under-weighted concept loss, not a real result. If results ever look like
#  nothing learned, check concept acc against chance FIRST.
#
#
#  COST
#  ----
#  ~130 s per model per 30 epochs on an M-series GPU. The default
#  (4 models x 5 seeds = 20 trainings) is roughly 45 minutes, less on a
#  cluster GPU. Well inside the 24 h wall time below.
#
#
#  RELATED FILES
#  -------------
#    scripts/exp_binned_concepts.py   the experiment this submits
#    cbmc/concepts/categorical.py     the Case 2/3 categorical block
#    cbmc/concepts/cem.py             the Case 3.5 block
#    run.sh                           the main experiment runner (different set)
#
# =============================================================================

#SBATCH --job-name=cbmc-binned
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

# --- 3. Execution ---
MAMBA_CMD="mamba run -p ./env"

usage () {
    cat <<EOF
Usage: sbatch run_binned.sh [options]

  --epochs N              training epochs per model            (default 30)
  --runs N                seeds per model; >1 gives mean+/-std (default 5)
  --bins "A B C"          bin counts to sweep                  (default "5 10 20")
  --concept-weight W      weight on the concept loss           (default 100)
                          NOTE: the task MSE spans 0-81 while the concept MSE
                          lives in [-1,1]. With W=1 the concept head is ~95x
                          under-weighted and concepts are NOT learned.
  --intervention-prob P   fraction of batches trained with true concepts
                                                               (default 0.25)
  --seed N                first seed                           (default 41)
  --tag NAME              suffix for the output CSV            (default none)
  --device D              cuda | mps | cpu                     (default auto)

Each option also reads an env var: EPOCHS, RUNS, BINS, CONCEPT_WEIGHT,
INTERVENTION_PROB, SEED, TAG, DEVICE.

Output: outputs/results/binned_concepts[_<tag>].csv
EOF
}

EPOCHS="${EPOCHS:-30}"
RUNS="${RUNS:-5}"
BINS="${BINS:-5 10 20}"
CONCEPT_WEIGHT="${CONCEPT_WEIGHT:-100}"
INTERVENTION_PROB="${INTERVENTION_PROB:-0.25}"
SEED="${SEED:-41}"
TAG="${TAG:-}"
DEVICE="${DEVICE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)              usage; exit 0 ;;
        --epochs)               EPOCHS="$2"; shift 2 ;;
        --runs)                 RUNS="$2"; shift 2 ;;
        --bins)                 BINS="$2"; shift 2 ;;
        --concept-weight)       CONCEPT_WEIGHT="$2"; shift 2 ;;
        --intervention-prob)    INTERVENTION_PROB="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --tag)                  TAG="$2"; shift 2 ;;
        --device)               DEVICE="$2"; shift 2 ;;
        --epochs=*)             EPOCHS="${1#*=}"; shift ;;
        --runs=*)               RUNS="${1#*=}"; shift ;;
        --bins=*)               BINS="${1#*=}"; shift ;;
        --concept-weight=*)     CONCEPT_WEIGHT="${1#*=}"; shift ;;
        --intervention-prob=*)  INTERVENTION_PROB="${1#*=}"; shift ;;
        --seed=*)               SEED="${1#*=}"; shift ;;
        --tag=*)                TAG="${1#*=}"; shift ;;
        --device=*)             DEVICE="${1#*=}"; shift ;;
        *) echo "ERROR: unknown option '$1'." >&2; usage >&2; exit 1 ;;
    esac
done

# --- validation: fail here, not 20 minutes into training ---
for pair in "EPOCHS:$EPOCHS" "RUNS:$RUNS" "SEED:$SEED"; do
    name="${pair%%:*}"; val="${pair#*:}"
    if ! [[ "$val" =~ ^[0-9]+$ ]] || [ "$val" -lt 1 ]; then
        echo "ERROR: $name must be an integer >= 1 (got '$val')." >&2; exit 1
    fi
done
for b in $BINS; do
    if ! [[ "$b" =~ ^[0-9]+$ ]] || [ "$b" -lt 2 ]; then
        echo "ERROR: each bin count must be an integer >= 2 (got '$b')." >&2; exit 1
    fi
done
if ! [[ "$CONCEPT_WEIGHT" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "ERROR: --concept-weight must be a number (got '$CONCEPT_WEIGHT')." >&2; exit 1
fi
if ! [[ "$INTERVENTION_PROB" =~ ^[0-9]*\.?[0-9]+$ ]] \
   || [ "$(awk -v p="$INTERVENTION_PROB" 'BEGIN{print (p<0||p>1)?1:0}')" -eq 1 ]; then
    echo "ERROR: --intervention-prob must be between 0 and 1 (got '$INTERVENTION_PROB')." >&2
    exit 1
fi

N_MODELS=$(( $(echo $BINS | wc -w) + 1 ))
echo "Epochs:            $EPOCHS"
echo "Runs (seeds):      $RUNS  (starting at $SEED)"
echo "Bin counts:        $BINS"
echo "Concept weight:    $CONCEPT_WEIGHT"
echo "Intervention prob: $INTERVENTION_PROB"
echo "Models:            $N_MODELS  ($N_MODELS x $RUNS = $((N_MODELS * RUNS)) trainings)"
echo ""

EXTRA=""
[ -n "$TAG" ]    && EXTRA="$EXTRA --tag $TAG"
[ -n "$DEVICE" ] && EXTRA="$EXTRA --device $DEVICE"

echo "+ python scripts/exp_binned_concepts.py --epochs $EPOCHS --runs $RUNS \\"
echo "    --bins $BINS --concept-weight $CONCEPT_WEIGHT \\"
echo "    --intervention-prob $INTERVENTION_PROB --seed $SEED$EXTRA"
echo ""

$MAMBA_CMD python scripts/exp_binned_concepts.py \
    --epochs            "$EPOCHS"            \
    --runs              "$RUNS"              \
    --bins              $BINS                \
    --concept-weight    "$CONCEPT_WEIGHT"    \
    --intervention-prob "$INTERVENTION_PROB" \
    --seed              "$SEED"              \
    $EXTRA

# --- 4. Cleanup ---
STATUS=$?
echo "========================================"
echo "End:         $(date)"
echo "Exit status: $STATUS"
echo "========================================"

exit $STATUS
