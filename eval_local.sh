#!/bin/bash
# =============================================================================
#  eval_local.sh — re-evaluate every saved benchmark model on this machine
# =============================================================================
#
#  1. copies the checkpoints from the cluster   (~1.2 GB, rsync: only new files)
#  2. renames old "cem_linear_raw" checkpoints that are really the no-LayerNorm
#     variant to "cem_linear_noln" (detected from the weights, safe to rerun)
#  3. runs scripts/run_benchmark.py --eval-only (no training) for every
#     dataset x model, with the raw-unit concept metrics
#  4. prints one summary table per dataset
#
#  Output
#    outputs/results/benchmark/<dataset>/<model>_eval.csv     one row per seed
#    outputs/results/benchmark/<dataset>_eval_summary.csv     mean / std
#
#  Usage
#    bash eval_local.sh                            # everything
#    bash eval_local.sh --no-sync                  # skip the download
#    bash eval_local.sh --datasets "mnist pendulum" --models "cem cem_linear"
#    bash eval_local.sh --overwrite                # re-evaluate seeds already done
#
#  Data loading runs in the main process (--num-workers 0): the dataset lambdas
#  cannot be pickled by macOS worker processes.
#
#  Resumable: seeds already in <model>_eval.csv are skipped (unless --overwrite).
#  Rough time on a laptop: mnist + pendulum a few minutes per model,
#  lidc ~5-10 min per model (30 test sets).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-/opt/miniconda3/envs/cbmc/bin/python}"
REMOTE="${REMOTE:-ouesly@piz:/home/ouesly/CBMC}"
DATASETS="mnist pendulum lidc"
MODELS="cbm cem cem_linear cem_linear_raw cem_linear_noln binned5 binned10 binned15"
SYNC=1
EXTRA=""

while [ $# -gt 0 ]; do
    case "$1" in
        --datasets)  DATASETS="$2"; shift 2 ;;
        --models)    MODELS="$2"; shift 2 ;;
        --no-sync)   SYNC=""; shift ;;
        --overwrite) EXTRA="--overwrite"; shift ;;
        -h|--help)   sed -n '2,/^# ====.*$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done

# --- 1. download ----------------------------------------------------------------
if [ -n "$SYNC" ]; then
    echo "== downloading checkpoints from $REMOTE"
    mkdir -p outputs/models
    rsync -a --info=progress2 "$REMOTE/outputs/models/benchmark" outputs/models/ \
        || { echo "rsync failed" >&2; exit 1; }
fi

# --- 2. old 'cem_linear_raw' (no LayerNorm) -> 'cem_linear_noln' ------------------
"$PY" - <<'PYEOF'
import glob, os, torch
n = 0
for f in glob.glob("outputs/models/benchmark/*/cem_linear_raw_seed*.pt"):
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    if not any(k.startswith("bott.output_norm.") for k in sd):       # no LayerNorm -> old variant
        new = f.replace("cem_linear_raw_seed", "cem_linear_noln_seed")
        if not os.path.exists(new):
            os.rename(f, new); n += 1
print(f"== renamed {n} old no-LayerNorm checkpoints to cem_linear_noln")
PYEOF

# --- 3. evaluate ------------------------------------------------------------------
for d in $DATASETS; do
    for m in $MODELS; do
        if ! ls outputs/models/benchmark/$d/${m}_seed*.pt >/dev/null 2>&1; then
            echo "== $d / $m: no checkpoints, skipped"; continue
        fi
        echo "== $d / $m"
        "$PY" scripts/run_benchmark.py --dataset "$d" --models "$m" --eval-only --tag eval \
              --num-workers 0 $EXTRA 2>&1 | grep -v -E "Warning|warn\(" | grep -E "seed [0-9]|skipped|Error|cannot" | tail -2
    done
done

# --- 4. summaries -----------------------------------------------------------------
for d in $DATASETS; do
    "$PY" scripts/run_benchmark.py --dataset "$d" --summarize --tag eval 2>&1 | grep -v -E "Warning|warn\("
done
